from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn

from ..jaxlayers import Conv1d
from .encoder import GroupNorm1D


def _hann_window(length: int, *, dtype: Any = jnp.float32) -> jnp.ndarray:
    if length <= 1:
        return jnp.ones((max(1, int(length)),), dtype=dtype)
    n = jnp.arange(int(length), dtype=dtype)
    return 0.5 - 0.5 * jnp.cos((2.0 * jnp.pi * n) / float(int(length) - 1))


def _swish(x: jnp.ndarray) -> jnp.ndarray:
    return x * nn.sigmoid(x)


def _overlap_add(frames: jnp.ndarray, *, hop_length: int) -> jnp.ndarray:
    """Sum frames (B, F, N) placed every `hop_length` samples -> (B, (F - 1) * hop + N)."""
    if frames.ndim != 3:
        raise ValueError(f"_overlap_add expects (B,F,N), got {frames.shape}")
    batch, num_frames, frame_length = frames.shape
    hop = int(hop_length)
    if frame_length % hop == 0:
        # Split each frame into hop-long segments and add them with shifts: no scatter, so it is
        # deterministic and cheaper (forward and backward) than the indexed add below.
        r_count = frame_length // hop
        seg = frames.reshape(batch, num_frames, r_count, hop)
        acc = None
        for r in range(r_count):
            part = jnp.pad(seg[:, :, r, :], ((0, 0), (r, r_count - 1 - r), (0, 0)))
            acc = part if acc is None else acc + part
        return acc.reshape(batch, (num_frames + r_count - 1) * hop)
    out_length = (int(num_frames) - 1) * int(hop_length) + int(frame_length)
    idx = (
        jnp.arange(int(num_frames), dtype=jnp.int32)[:, None] * int(hop_length)
        + jnp.arange(int(frame_length), dtype=jnp.int32)[None, :]
    ).reshape((-1,))

    def _one(frames_one: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros((out_length,), dtype=frames_one.dtype).at[idx].add(frames_one.reshape((-1,)))

    return jax.vmap(_one)(frames)


def _istft_same(spec: jnp.ndarray, *, n_fft: int, hop_length: int) -> jnp.ndarray:
    """ISTFT with Vocos-style same padding/cropping for (B, frames, bins) complex specs."""
    if spec.ndim != 3:
        raise ValueError(f"ISTFT spec must be (B,F,N), got {spec.shape}")
    num_frames = int(spec.shape[1])
    target_length = num_frames * int(hop_length)
    window = _hann_window(int(n_fft), dtype=jnp.float32)
    frames = jnp.fft.irfft(spec, n=int(n_fft), axis=-1).astype(jnp.float32)
    frames = frames * window[None, None, :]
    y = _overlap_add(frames, hop_length=int(hop_length))
    envelope = _overlap_add(
        jnp.broadcast_to(jnp.square(window)[None, None, :], (1, num_frames, int(n_fft))),
        hop_length=int(hop_length),
    )[0]
    crop_total = int(y.shape[-1]) - int(target_length)
    crop_left = max(0, crop_total // 2)
    y = y[:, crop_left : crop_left + target_length]
    envelope = envelope[crop_left : crop_left + target_length]
    return y / jnp.maximum(envelope[None, :], 1e-8)


class ConvNeXtBlock1D(nn.Module):
    dim: int
    intermediate_dim: int
    layer_scale_init_value: float = 1.0 / 12.0
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    def setup(self) -> None:
        self.dwconv = Conv1d(
            self.dim,
            kernel=7,
            padding="REFLECT",
            feature_group_count=self.dim,
            use_bias=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="dwconv",
        )
        self.norm = nn.LayerNorm(dtype=self.dtype, param_dtype=self.param_dtype, name="norm")
        self.pwconv1 = nn.Dense(
            self.intermediate_dim,
            use_bias=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="pwconv1",
        )
        self.pwconv2 = nn.Dense(
            self.dim,
            use_bias=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="pwconv2",
        )
        self.gamma = self.param(
            "gamma",
            nn.initializers.constant(float(self.layer_scale_init_value)),
            (self.dim,),
            self.param_dtype,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = self.dwconv(x)
        h = self.norm(h)
        h = self.pwconv1(h)
        h = nn.gelu(h, approximate=True)
        h = self.pwconv2(h)
        h = h * self.gamma.astype(h.dtype)
        return x + h


class DecoderResnetBlock1D(nn.Module):
    dim: int
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    def setup(self) -> None:
        self.norm1 = GroupNorm1D(self.dim, dtype=self.dtype, param_dtype=self.param_dtype, name="norm1")
        self.conv1 = Conv1d(
            self.dim,
            kernel=3,
            padding="REFLECT",
            use_bias=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="conv1",
        )
        self.norm2 = GroupNorm1D(self.dim, dtype=self.dtype, param_dtype=self.param_dtype, name="norm2")
        self.conv2 = Conv1d(
            self.dim,
            kernel=3,
            padding="REFLECT",
            use_bias=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="conv2",
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = self.norm1(x)
        h = _swish(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = _swish(h)
        h = self.conv2(h)
        return x + h


class DecoderSelfAttentionBlock1D(nn.Module):
    dim: int
    num_heads: int = 12
    attention_backend: str = "jax_cudnn"
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    def setup(self) -> None:
        if self.dim % self.num_heads != 0:
            raise ValueError(f"dim={self.dim} must be divisible by num_heads={self.num_heads}")
        backend = str(self.attention_backend).strip().lower()
        aliases = {
            "flash": "jax_cudnn",
            "flash_attention": "jax_cudnn",
            "cudnn": "jax_cudnn",
            "jax_cudnn": "jax_cudnn",
            "xla": "xla",
            "auto": "auto",
        }
        if backend not in aliases:
            raise ValueError(
                f"Unsupported decoder pos_net attention_backend={self.attention_backend!r}; "
                "choose from ['jax_cudnn', 'flash', 'xla', 'auto']."
            )
        self._attention_backend = aliases[backend]
        self._head_dim = int(self.dim) // int(self.num_heads)
        self.norm = GroupNorm1D(self.dim, dtype=self.dtype, param_dtype=self.param_dtype, name="norm")
        self.q = Conv1d(
            self.dim,
            kernel=1,
            padding="REFLECT",
            use_bias=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="q",
        )
        self.k = Conv1d(
            self.dim,
            kernel=1,
            padding="REFLECT",
            use_bias=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="k",
        )
        self.v = Conv1d(
            self.dim,
            kernel=1,
            padding="REFLECT",
            use_bias=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="v",
        )
        self.proj_out = Conv1d(
            self.dim,
            kernel=1,
            padding="REFLECT",
            use_bias=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="proj_out",
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if x.ndim != 3:
            raise ValueError(f"DecoderSelfAttentionBlock1D expects (B,T,C), got {x.shape}")
        h = self.norm(x)
        batch, seq_len, _ = h.shape
        q = self.q(h).reshape(batch, seq_len, int(self.num_heads), int(self._head_dim))
        k = self.k(h).reshape(batch, seq_len, int(self.num_heads), int(self._head_dim))
        v = self.v(h).reshape(batch, seq_len, int(self.num_heads), int(self._head_dim))
        if self._attention_backend == "jax_cudnn":
            implementation = "cudnn"
            qkv_dtype = jnp.bfloat16
        elif self._attention_backend == "xla":
            implementation = "xla"
            qkv_dtype = jnp.float32
        else:
            implementation = None
            qkv_dtype = jnp.float32
        h = jax.nn.dot_product_attention(
            q.astype(qkv_dtype),
            k.astype(qkv_dtype),
            v.astype(qkv_dtype),
            implementation=implementation,
        )
        h = h.reshape(batch, seq_len, int(self.dim)).astype(self.dtype)
        h = self.proj_out(h)
        return x + h


class DecoderPosNet1D(nn.Module):
    dim: int
    attention_heads: int = 12
    attention_backend: str = "jax_cudnn"
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    def setup(self) -> None:
        self.resnet_0 = DecoderResnetBlock1D(
            dim=self.dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="resnet_0",
        )
        self.resnet_1 = DecoderResnetBlock1D(
            dim=self.dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="resnet_1",
        )
        self.attn = DecoderSelfAttentionBlock1D(
            dim=self.dim,
            num_heads=self.attention_heads,
            attention_backend=self.attention_backend,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="attn",
        )
        self.resnet_2 = DecoderResnetBlock1D(
            dim=self.dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="resnet_2",
        )
        self.resnet_3 = DecoderResnetBlock1D(
            dim=self.dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="resnet_3",
        )
        self.norm = GroupNorm1D(self.dim, dtype=self.dtype, param_dtype=self.param_dtype, name="norm")

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = self.resnet_0(x)
        h = self.resnet_1(h)
        h = self.attn(h)
        h = self.resnet_2(h)
        h = self.resnet_3(h)
        return self.norm(h)


class ISTFTDecoder1D(nn.Module):
    """Vocos-style decoder: a ConvNeXt backbone predicts each frame's complex spectrum
    (real and imaginary parts of n_fft // 2 + 1 bins) and an inverse STFT with hop_length turns
    the frames into the waveform."""

    out_channels: int = 1
    input_dim: int = 512
    dim: int = 512
    intermediate_dim: int = 1536
    num_layers: int = 12
    pos_net_enabled: bool = True
    pos_net_attention_heads: int = 12
    pos_net_attention_backend: str = "jax_cudnn"
    n_fft: int = 64
    hop_length: int = 16
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    def setup(self) -> None:
        if self.out_channels != 1:
            raise ValueError("ISTFTDecoder1D supports one output channel.")
        if self.n_fft <= 0 or self.hop_length <= 0:
            raise ValueError("n_fft and hop_length must be positive.")
        if self.n_fft < self.hop_length:
            raise ValueError("n_fft should be >= hop_length for overlap-add synthesis.")
        self.embed = Conv1d(
            self.dim,
            kernel=7,
            padding="REFLECT",
            use_bias=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="embed",
        )
        if self.pos_net_enabled:
            self.pos_net = DecoderPosNet1D(
                dim=self.dim,
                attention_heads=int(self.pos_net_attention_heads),
                attention_backend=str(self.pos_net_attention_backend),
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="pos_net",
            )
        else:
            self.pos_net = None
        layer_scale = 1.0 / max(1, int(self.num_layers))
        self.blocks = tuple(
            ConvNeXtBlock1D(
                dim=self.dim,
                intermediate_dim=self.intermediate_dim,
                layer_scale_init_value=layer_scale,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name=f"convnext_{idx}",
            )
            for idx in range(max(0, int(self.num_layers)))
        )
        self.final_norm = nn.LayerNorm(dtype=self.dtype, param_dtype=self.param_dtype, name="final_norm")
        self.coeff_proj = Conv1d(
            (int(self.n_fft) // 2 + 1) * 2,
            kernel=1,
            padding="REFLECT",
            use_bias=True,
            dtype=jnp.float32,
            param_dtype=self.param_dtype,
            name="coeff_proj",
        )

    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        h = self.embed(z.astype(self.dtype))
        if self.pos_net is not None:
            h = self.pos_net(h)
        for block in self.blocks:
            h = block(h)
        h = self.final_norm(h).astype(jnp.float32)
        bins = int(self.n_fft) // 2 + 1
        coeff = self.coeff_proj(h).reshape(h.shape[0], h.shape[1], bins, 2)
        real = coeff[..., 0]
        # The DC and Nyquist bins of a real signal have no imaginary part.
        imag = coeff[..., 1].at[..., 0].set(0.0).at[..., -1].set(0.0)
        spec = real.astype(jnp.complex64) + 1j * imag.astype(jnp.complex64)
        return _istft_same(spec, n_fft=int(self.n_fft), hop_length=int(self.hop_length))
