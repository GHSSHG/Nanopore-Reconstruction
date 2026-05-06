from __future__ import annotations

import math
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from ..jaxlayers import Conv1d
from .encoder import GroupNorm1D


def _hann_window(length: int, *, dtype: Any = jnp.float32) -> jnp.ndarray:
    if length <= 1:
        return jnp.ones((max(1, int(length)),), dtype=dtype)
    n = jnp.arange(int(length), dtype=dtype)
    return 0.5 - 0.5 * jnp.cos((2.0 * jnp.pi * n) / float(int(length) - 1))


def _band_ids_from_edges(*, n_fft: int, sample_rate: float, edges_hz: Sequence[float]) -> tuple[int, ...]:
    bins = int(n_fft) // 2 + 1
    bin_hz = float(sample_rate) / float(n_fft)
    stops = [min(bins, max(1, int(math.floor(float(edge) / bin_hz)) + 1)) for edge in edges_hz]
    ids = []
    for idx in range(bins):
        band = 0
        for stop in stops:
            if idx < int(stop):
                break
            band += 1
        ids.append(band)
    return tuple(ids)


def _dropout(x: jnp.ndarray, *, rate: float, train: bool, rng: jax.Array | None) -> jnp.ndarray:
    if (not train) or rate <= 0.0:
        return x
    if rng is None:
        raise ValueError("dropout requires rng when train=True")
    keep_prob = 1.0 - float(rate)
    mask = jax.random.bernoulli(rng, p=keep_prob, shape=x.shape)
    return jnp.where(mask, x / keep_prob, 0.0)


def _swish(x: jnp.ndarray) -> jnp.ndarray:
    return x * nn.sigmoid(x)


def _overlap_add(frames: jnp.ndarray, *, hop_length: int) -> jnp.ndarray:
    if frames.ndim != 3:
        raise ValueError(f"_overlap_add expects (B,F,N), got {frames.shape}")
    _, num_frames, frame_length = frames.shape
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

    def __call__(self, x: jnp.ndarray, *, train: bool = False) -> jnp.ndarray:
        del train
        h = self.dwconv(x)
        h = self.norm(h)
        h = self.pwconv1(h)
        h = nn.gelu(h, approximate=True)
        h = self.pwconv2(h)
        h = h * self.gamma.astype(h.dtype)
        return x + h


class DecoderResnetBlock1D(nn.Module):
    dim: int
    dropout: float = 0.0
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

    def __call__(self, x: jnp.ndarray, *, train: bool = False, rng: jax.Array | None = None) -> jnp.ndarray:
        h = self.norm1(x)
        h = _swish(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = _swish(h)
        h = _dropout(h, rate=float(self.dropout), train=train, rng=rng)
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

    def __call__(self, x: jnp.ndarray, *, train: bool = False) -> jnp.ndarray:
        del train
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
    dropout: float = 0.0
    attention_heads: int = 12
    attention_backend: str = "jax_cudnn"
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    def setup(self) -> None:
        self.resnet_0 = DecoderResnetBlock1D(
            dim=self.dim,
            dropout=self.dropout,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="resnet_0",
        )
        self.resnet_1 = DecoderResnetBlock1D(
            dim=self.dim,
            dropout=self.dropout,
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
            dropout=self.dropout,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="resnet_2",
        )
        self.resnet_3 = DecoderResnetBlock1D(
            dim=self.dim,
            dropout=self.dropout,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="resnet_3",
        )
        self.norm = GroupNorm1D(self.dim, dtype=self.dtype, param_dtype=self.param_dtype, name="norm")

    def __call__(self, x: jnp.ndarray, *, train: bool = False, rng: jax.Array | None = None) -> jnp.ndarray:
        block_rngs: list[jax.Array | None] = [None, None, None, None]
        if rng is not None:
            block_rngs = list(jax.random.split(rng, 4))
        h = self.resnet_0(x, train=train, rng=block_rngs[0])
        h = self.resnet_1(h, train=train, rng=block_rngs[1])
        h = self.attn(h, train=train)
        h = self.resnet_2(h, train=train, rng=block_rngs[2])
        h = self.resnet_3(h, train=train, rng=block_rngs[3])
        return self.norm(h)


class BandedGatedISTFTDecoder1D(nn.Module):
    out_channels: int = 1
    input_dim: int = 512
    dim: int = 512
    intermediate_dim: int = 1536
    num_layers: int = 12
    pos_net_enabled: bool = True
    pos_net_dropout: float = 0.0
    pos_net_attention_heads: int = 12
    pos_net_attention_backend: str = "jax_cudnn"
    n_fft: int = 512
    hop_length: int = 24
    sample_rate: float = 5000.0
    band_edges_hz: Sequence[float] = (200.0, 500.0, 1000.0)
    band_gain_init: Sequence[float] = (1.0, 0.8, 0.45, 0.12)
    dynamic_gate_scale: float = 0.5
    residual_enabled: bool = True
    residual_hidden_dim: int = 128
    residual_alpha_init: float = 0.0
    residual_alpha_max: float = 0.1
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    def setup(self) -> None:
        if self.out_channels != 1:
            raise ValueError("BandedGatedISTFTDecoder1D currently supports one output channel.")
        if self.n_fft <= 0 or self.hop_length <= 0:
            raise ValueError("n_fft and hop_length must be positive.")
        if self.n_fft < self.hop_length:
            raise ValueError("n_fft should be >= hop_length for overlap-add synthesis.")
        edges_hz = tuple(float(v) for v in self.band_edges_hz)
        nyquist_hz = float(self.sample_rate) / 2.0
        if any(edge <= 0.0 or edge >= nyquist_hz for edge in edges_hz):
            raise ValueError(f"band_edges_hz must be within (0, Nyquist={nyquist_hz}), got {edges_hz}.")
        if any(right <= left for left, right in zip(edges_hz, edges_hz[1:])):
            raise ValueError(f"band_edges_hz must be strictly increasing, got {edges_hz}.")
        band_gain_init = tuple(float(v) for v in self.band_gain_init)
        num_bands = len(edges_hz) + 1
        if len(band_gain_init) != num_bands or any(v <= 0 for v in band_gain_init):
            raise ValueError(f"band_gain_init must contain {num_bands} positive values.")
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
                dropout=float(self.pos_net_dropout),
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
        bins = int(self.n_fft) // 2 + 1
        self.coeff_proj = Conv1d(
            bins * 2,
            kernel=1,
            padding="REFLECT",
            use_bias=True,
            dtype=jnp.float32,
            param_dtype=self.param_dtype,
            name="coeff_proj",
        )
        self.gate_proj = Conv1d(
            num_bands,
            kernel=1,
            padding="REFLECT",
            use_bias=True,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            dtype=jnp.float32,
            param_dtype=self.param_dtype,
            name="gate_proj",
        )
        if self.residual_enabled:
            self.residual_proj1 = Conv1d(
                self.residual_hidden_dim,
                kernel=1,
                padding="REFLECT",
                use_bias=True,
                dtype=jnp.float32,
                param_dtype=self.param_dtype,
                name="residual_proj1",
            )
            self.residual_proj2 = Conv1d(
                int(self.hop_length),
                kernel=1,
                padding="REFLECT",
                use_bias=True,
                dtype=jnp.float32,
                param_dtype=self.param_dtype,
                name="residual_proj2",
            )
        else:
            self.residual_proj1 = None
            self.residual_proj2 = None
        self._band_ids = _band_ids_from_edges(
            n_fft=int(self.n_fft),
            sample_rate=float(self.sample_rate),
            edges_hz=edges_hz,
        )
        init = jnp.log(jnp.asarray(band_gain_init, dtype=jnp.float32))
        self.log_band_gain = self.param(
            "log_band_gain",
            lambda _rng, _shape, _dtype=None: init,
            (num_bands,),
            self.param_dtype,
        )
        alpha_max = max(1e-8, float(self.residual_alpha_max))
        alpha_init = min(max(float(self.residual_alpha_init), 0.0), alpha_max)
        p = np.clip(alpha_init / alpha_max if alpha_init > 0 else 1e-3, 1e-6, 1.0 - 1e-6)
        alpha_raw_init = float(math.log(p / (1.0 - p)))
        self.residual_raw_alpha = self.param(
            "residual_raw_alpha",
            nn.initializers.constant(alpha_raw_init),
            (),
            self.param_dtype,
        )

    def _band_gains(self) -> jnp.ndarray:
        return jnp.exp(self.log_band_gain.astype(jnp.float32))

    def _band_gain(self) -> jnp.ndarray:
        band_ids = jnp.asarray(self._band_ids, dtype=jnp.int32)
        return self._band_gains()[band_ids]

    def _residual_alpha(self) -> jnp.ndarray:
        return float(self.residual_alpha_max) * nn.sigmoid(self.residual_raw_alpha.astype(jnp.float32))

    def __call__(self, z: jnp.ndarray, *, train: bool = False, rng: jax.Array | None = None):
        pos_rng = None
        if rng is not None:
            pos_rng, _ = jax.random.split(rng)
        h = z.astype(self.dtype)
        h = self.embed(h)
        if self.pos_net is not None:
            h = self.pos_net(h, train=train, rng=pos_rng)
        for block in self.blocks:
            h = block(h, train=train)
        h = self.final_norm(h)
        h_fp32 = h.astype(jnp.float32)

        bins = int(self.n_fft) // 2 + 1
        coeff_raw = self.coeff_proj(h_fp32).reshape(h_fp32.shape[0], h_fp32.shape[1], bins, 2)
        gate_raw = self.gate_proj(h_fp32)
        gates = 1.0 + float(self.dynamic_gate_scale) * jnp.tanh(gate_raw.astype(jnp.float32))
        band_ids = jnp.asarray(self._band_ids, dtype=jnp.int32)
        gate_by_bin = jnp.take(gates, band_ids, axis=-1)
        band_gains = self._band_gains()
        gain_by_bin = band_gains[band_ids]
        coeff = coeff_raw * gain_by_bin[None, None, :, None] * gate_by_bin[..., None]
        real = coeff[..., 0]
        imag = coeff[..., 1]
        imag = imag.at[..., 0].set(0.0)
        imag = imag.at[..., -1].set(0.0)
        spec = real.astype(jnp.complex64) + (1j * imag.astype(jnp.complex64))
        wave_istft = _istft_same(spec, n_fft=int(self.n_fft), hop_length=int(self.hop_length))

        if self.residual_enabled:
            residual = self.residual_proj1(h_fp32)
            residual = nn.gelu(residual, approximate=True)
            residual = self.residual_proj2(residual)
            residual = residual.reshape((residual.shape[0], residual.shape[1] * int(self.hop_length)))
            alpha = self._residual_alpha()
        else:
            residual = jnp.zeros_like(wave_istft)
            alpha = jnp.asarray(0.0, dtype=jnp.float32)
        wave = wave_istft + alpha * residual

        energy = jnp.square(real) + jnp.square(imag)
        num_bands = int(self.log_band_gain.shape[0])
        band_masks = tuple((band_ids == idx).astype(jnp.float32) for idx in range(num_bands))

        def _band_mean(mask: jnp.ndarray) -> jnp.ndarray:
            denom = jnp.maximum(jnp.sum(mask), 1.0)
            return jnp.sum(energy * mask[None, None, :]) / (energy.shape[0] * energy.shape[1] * denom)

        def _band_sum(mask: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum(energy * mask[None, None, :])

        wave_rms = jnp.sqrt(jnp.mean(jnp.square(wave_istft)) + 1e-8)
        residual_rms = jnp.sqrt(jnp.mean(jnp.square(residual)) + 1e-8)
        total_energy = jnp.sum(energy) + 1e-8
        band_energy_means = tuple(_band_mean(mask) for mask in band_masks)
        band_energy_ratios = tuple(_band_sum(mask) / total_energy for mask in band_masks)
        residual_spec = jnp.fft.rfft(residual, axis=-1)
        freqs = jnp.fft.rfftfreq(residual.shape[-1], d=1.0 / float(self.sample_rate))
        high_res_mask = (freqs > 1000.0).astype(jnp.float32)
        residual_high_energy = jnp.sum(jnp.square(jnp.abs(residual_spec)) * high_res_mask[None, :])
        residual_total_energy = jnp.sum(jnp.square(jnp.abs(residual_spec))) + 1e-8
        aux = {
            "istft_low_band_energy": band_energy_means[0],
            "istft_mid_band_energy": band_energy_means[min(1, num_bands - 1)],
            "istft_high_band_energy": band_energy_means[-1],
            "istft_high_band_ratio": band_energy_ratios[-1],
            "istft_static_gain_low": band_gains[0],
            "istft_static_gain_mid": band_gains[min(1, num_bands - 1)],
            "istft_static_gain_high": band_gains[-1],
            "istft_gate_low_mean": jnp.mean(gates[..., 0]),
            "istft_gate_mid_mean": jnp.mean(gates[..., min(1, num_bands - 1)]),
            "istft_gate_high_mean": jnp.mean(gates[..., -1]),
            "residual_alpha": alpha,
            "residual_energy_l2_raw": jnp.mean(jnp.square(residual)),
            "residual_rms": residual_rms,
            "wave_istft_rms": wave_rms,
            "residual_rms_ratio": residual_rms / wave_rms,
            "residual_high_band_ratio": residual_high_energy / residual_total_energy,
        }
        for idx in range(num_bands):
            aux[f"istft_band{idx}_energy"] = band_energy_means[idx]
            aux[f"istft_band{idx}_ratio"] = band_energy_ratios[idx]
            aux[f"istft_static_gain_band{idx}"] = band_gains[idx]
            aux[f"istft_gate_band{idx}_mean"] = jnp.mean(gates[..., idx])
        return wave.astype(jnp.float32), aux
