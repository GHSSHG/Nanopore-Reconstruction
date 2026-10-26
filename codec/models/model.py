from __future__ import annotations

from typing import Any, Dict, Tuple

import jax.numpy as jnp
from flax import linen as nn

from .decoder import ISTFTDecoder1D
from .encoder import SimVQEncoder1D
from .quantize import SimVQ1D
from .recurrent import ResidualBiLSTM1D
from ..jaxlayers import Conv1d


class SimVQAudioModel(nn.Module):
    in_channels: int = 1
    enc_channels: Tuple[int, ...] = (32, 64, 128, 256, 512)
    enc_num_res_blocks: int = 4
    enc_stage_num_res_blocks: Tuple[int, ...] | None = (2, 2, 3, 3)
    enc_down_strides: Tuple[int, ...] = (2, 2, 2, 3)
    latent_dim: int = 512
    quantizer_dim: int | None = 512
    codebook_size: int = 16384
    quantizer_l2_normalize: bool = False
    quantizer_learn_codebook: bool = False
    dec_channels: Tuple[int, ...] = (512,)
    enc_kernel_size: int = 7
    enc_dtype: Any = jnp.float32
    dec_dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    latent_bilstm_layers: int = 2
    latent_bilstm_hidden_dim: int | None = 256
    quant_conv_kernel_size: int = 7
    post_quant_conv_kernel_size: int = 7
    encoder_use_block_norm: bool = True
    encoder_use_input_norm: bool = True
    encoder_use_transition_norm: bool = True
    decoder_dim: int = 512
    decoder_intermediate_dim: int = 1536
    decoder_num_layers: int = 12
    decoder_pos_net_enabled: bool = True
    decoder_pos_net_attention_heads: int = 12
    decoder_pos_net_attention_backend: str = "jax_cudnn"
    istft_n_fft: int = 64
    istft_hop_length: int = 16

    def setup(self) -> None:
        enc_channels = tuple(int(ch) for ch in self.enc_channels)
        dec_channels = tuple(int(ch) for ch in self.dec_channels)
        enc_stage_num_res_blocks = (
            tuple(int(v) for v in self.enc_stage_num_res_blocks)
            if self.enc_stage_num_res_blocks is not None
            else self.enc_num_res_blocks
        )
        if len(enc_channels) != len(self.enc_down_strides) + 1:
            raise ValueError("enc_channels must be one longer than enc_down_strides")
        if not dec_channels:
            raise ValueError("dec_channels must contain the decoder input channel count")
        if enc_channels[-1] != self.latent_dim:
            raise ValueError(
                f"Encoder output channels ({enc_channels[-1]}) must match latent_dim ({self.latent_dim})"
            )
        if dec_channels[0] != self.latent_dim:
            raise ValueError(
                f"Decoder input channels ({dec_channels[0]}) must match latent_dim ({self.latent_dim})"
            )
        quantizer_dim = self.latent_dim if self.quantizer_dim is None else int(self.quantizer_dim)
        if quantizer_dim <= 0:
            raise ValueError(f"quantizer_dim must be positive, got {quantizer_dim}")

        self.encoder = SimVQEncoder1D(
            in_channels=self.in_channels,
            channels=enc_channels,
            num_res_blocks=enc_stage_num_res_blocks,
            down_strides=self.enc_down_strides,
            input_kernel_size=self.enc_kernel_size,
            use_block_norm=self.encoder_use_block_norm,
            use_input_norm=self.encoder_use_input_norm,
            use_transition_norm=self.encoder_use_transition_norm,
            dtype=self.enc_dtype,
            param_dtype=self.param_dtype,
        )
        self.decoder = ISTFTDecoder1D(
            out_channels=self.in_channels,
            input_dim=dec_channels[0],
            dim=int(self.decoder_dim),
            intermediate_dim=int(self.decoder_intermediate_dim),
            num_layers=int(self.decoder_num_layers),
            pos_net_enabled=bool(self.decoder_pos_net_enabled),
            pos_net_attention_heads=int(self.decoder_pos_net_attention_heads),
            pos_net_attention_backend=str(self.decoder_pos_net_attention_backend),
            n_fft=int(self.istft_n_fft),
            hop_length=int(self.istft_hop_length),
            dtype=self.dec_dtype,
            param_dtype=self.param_dtype,
        )

        quant_path_dtype = jnp.float32
        self.quant_conv = Conv1d(
            quantizer_dim,
            kernel=int(self.quant_conv_kernel_size),
            use_bias=False,
            padding="REFLECT",
            dtype=quant_path_dtype,
            param_dtype=quant_path_dtype,
            name="quant_conv",
        )
        self.post_quant_conv = Conv1d(
            dec_channels[0],
            kernel=int(self.post_quant_conv_kernel_size),
            use_bias=False,
            padding="REFLECT",
            dtype=quant_path_dtype,
            param_dtype=quant_path_dtype,
            name="post_quant_conv",
        )
        self.quantizer = SimVQ1D(
            codebook_size=self.codebook_size,
            code_dim=quantizer_dim,
            l2_normalize=bool(self.quantizer_l2_normalize),
            learn_codebook=bool(self.quantizer_learn_codebook),
        )
        if int(self.latent_bilstm_layers) > 0:
            self.latent_bilstm = ResidualBiLSTM1D(
                dim=int(self.latent_dim),
                num_layers=int(self.latent_bilstm_layers),
                hidden_dim=self.latent_bilstm_hidden_dim,
                dtype=self.enc_dtype,
                param_dtype=self.param_dtype,
                name="latent_bilstm",
            )
        else:
            self.latent_bilstm = None

    def latents(self, x: jnp.ndarray) -> jnp.ndarray:
        """Normalized chunks (B, L) -> pre-quantization latents (B, T, D)."""
        h_e = self.encoder(x).astype(jnp.float32)
        if self.latent_bilstm is not None:
            h_e = self.latent_bilstm(h_e.astype(self.enc_dtype)).astype(jnp.float32)
        return self.quant_conv(h_e.astype(jnp.float32))

    def encode(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
        h_qin = self.latents(x)
        z_q, info = self.quantizer(h_qin)
        return h_qin, z_q.astype(jnp.float32), info

    def init_forward(self, x: jnp.ndarray):
        """For model.init only: touches every submodule without the GPU codebook search."""
        return self.decode(self.latents(x)), self.quantizer.projected_codebook()

    def decode(self, z_q: jnp.ndarray) -> jnp.ndarray:
        z_dec = self.post_quant_conv(z_q.astype(jnp.float32))
        return self.decoder(z_dec.astype(jnp.float32)).astype(jnp.float32)

    def __call__(self, x: jnp.ndarray) -> Dict[str, Any]:
        z_e, z_q, info = self.encode(x)
        return {"wave_hat": self.decode(z_q), "enc": {"z_e": z_e, "z_q": z_q, **info}}
