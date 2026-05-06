from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

from .decoder import BandedGatedISTFTDecoder1D
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
    dec_channels: Tuple[int, ...] = (512,)
    enc_kernel_size: int = 7
    enc_dtype: Any = jnp.float32
    dec_dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    latent_bilstm_layers: int = 2
    latent_bilstm_hidden_dim: int | None = 256
    diveq_sigma2: float = 1e-3
    search_chunk_size: int = 2048
    quant_conv_kernel_size: int = 7
    post_quant_conv_kernel_size: int = 7
    encoder_use_block_norm: bool = True
    encoder_use_input_norm: bool = True
    encoder_use_transition_norm: bool = True
    decoder_dim: int = 512
    decoder_intermediate_dim: int = 1536
    decoder_num_layers: int = 12
    decoder_pos_net_enabled: bool = True
    decoder_pos_net_dropout: float = 0.0
    decoder_pos_net_attention_heads: int = 12
    decoder_pos_net_attention_backend: str = "jax_cudnn"
    istft_n_fft: int = 512
    istft_hop_length: int = 24
    istft_sample_rate: float = 5000.0
    istft_band_edges_hz: Tuple[float, ...] = (200.0, 500.0, 1000.0)
    istft_band_gain_init: Tuple[float, ...] = (1.0, 0.8, 0.45, 0.12)
    istft_dynamic_gate_scale: float = 0.5
    residual_correction_enabled: bool = False
    residual_correction_alpha_init: float = 0.0
    residual_correction_alpha_max: float = 0.1
    residual_correction_hidden_dim: int = 128

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
        self.decoder = BandedGatedISTFTDecoder1D(
            out_channels=self.in_channels,
            input_dim=dec_channels[0],
            dim=int(self.decoder_dim),
            intermediate_dim=int(self.decoder_intermediate_dim),
            num_layers=int(self.decoder_num_layers),
            pos_net_enabled=bool(self.decoder_pos_net_enabled),
            pos_net_dropout=float(self.decoder_pos_net_dropout),
            pos_net_attention_heads=int(self.decoder_pos_net_attention_heads),
            pos_net_attention_backend=str(self.decoder_pos_net_attention_backend),
            n_fft=int(self.istft_n_fft),
            hop_length=int(self.istft_hop_length),
            sample_rate=float(self.istft_sample_rate),
            band_edges_hz=tuple(float(v) for v in self.istft_band_edges_hz),
            band_gain_init=tuple(float(v) for v in self.istft_band_gain_init),
            dynamic_gate_scale=float(self.istft_dynamic_gate_scale),
            residual_enabled=bool(self.residual_correction_enabled),
            residual_hidden_dim=int(self.residual_correction_hidden_dim),
            residual_alpha_init=float(self.residual_correction_alpha_init),
            residual_alpha_max=float(self.residual_correction_alpha_max),
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
            diveq_sigma2=self.diveq_sigma2,
            search_chunk_size=max(1, int(self.search_chunk_size)),
            dtype=quant_path_dtype,
            param_dtype=quant_path_dtype,
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

    def encode(
        self,
        x: jnp.ndarray,
        *,
        train: bool = False,
        offset: int = 0,
        rng: jax.random.KeyArray,
        collect_codebook_stats: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
        enc_rng, _, quant_rng = jax.random.split(rng, 3)
        h_e = self.encoder(x, train=train, offset=offset, rng=enc_rng).astype(jnp.float32)
        if self.latent_bilstm is not None:
            h_e = self.latent_bilstm(h_e.astype(self.enc_dtype), train=train).astype(jnp.float32)
        h_qin = self.quant_conv(h_e.astype(jnp.float32))
        z_q, info = self.quantizer(
            h_qin.astype(jnp.float32),
            rng=quant_rng,
            train=train,
            collect_codebook_stats=collect_codebook_stats,
        )
        return h_qin, z_q.astype(jnp.float32), info

    def decode(self, z_q: jnp.ndarray, *, train: bool = False, rng: jax.random.KeyArray | None = None):
        dec_rng = None
        if rng is not None:
            _, dec_rng = jax.random.split(rng)
        z_dec = self.post_quant_conv(z_q.astype(jnp.float32))
        wave, aux = self.decoder(z_dec.astype(jnp.float32), train=train, rng=dec_rng)
        if wave.ndim == 3 and wave.shape[-1] == 1:
            wave = jnp.squeeze(wave, axis=-1)
        return wave.astype(jnp.float32), aux

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        train: bool = False,
        offset: int = 0,
        rng: jax.random.KeyArray,
        collect_codebook_stats: bool = True,
    ) -> Dict[str, Any]:
        _, enc_rng, dec_rng = jax.random.split(rng, 3)
        z_e, z_q, info = self.encode(
            x,
            train=train,
            offset=offset,
            rng=enc_rng,
            collect_codebook_stats=collect_codebook_stats,
        )
        wave_hat, dec_aux = self.decode(z_q, train=train, rng=dec_rng)
        usage_ratio = info.get("usage_ratio", jnp.array(0.0, dtype=z_e.dtype))
        return {
            "wave_hat": wave_hat,
            "enc": {
                "z_e": z_e,
                "z_q": z_q,
                "indices": info["indices"],
                "perplexity": info["perplexity"],
                "usage_ratio": usage_ratio,
                "code_usage": usage_ratio,
                "q_z_dist": info.get("q_z_dist", jnp.array(0.0, dtype=z_e.dtype)),
                "log_q_z_dist": info.get("log_q_z_dist", jnp.array(0.0, dtype=z_e.dtype)),
                "token_counts": info.get("token_counts"),
                "total_tokens": info.get("total_tokens"),
            },
            "dec": dec_aux,
        }
