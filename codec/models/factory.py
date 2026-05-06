from __future__ import annotations

from typing import Any, Sequence

import jax.numpy as jnp

from .model import SimVQAudioModel


def _resolve_dtype(dtype_value: Any, *, fallback: Any = jnp.float32) -> Any:
    if dtype_value is None:
        return fallback
    if isinstance(dtype_value, str):
        key = dtype_value.strip().lower()
        mapping = {
            "fp32": jnp.float32,
            "float32": jnp.float32,
            "bf16": jnp.bfloat16,
            "bfloat16": jnp.bfloat16,
            "fp16": jnp.float16,
            "float16": jnp.float16,
        }
        if key not in mapping:
            raise ValueError(f"Unsupported dtype {dtype_value}.")
        return mapping[key]
    return dtype_value


def _tuple_cfg(model_cfg: dict[str, Any], key: str, default: Sequence[int]) -> tuple[int, ...]:
    value = model_cfg.get(key, default)
    return tuple(int(v) for v in value)


def _optional_tuple_cfg(model_cfg: dict[str, Any], key: str) -> tuple[int, ...] | None:
    if key not in model_cfg or model_cfg.get(key) is None:
        return None
    return tuple(int(v) for v in model_cfg[key])


def _normalize_variant(raw_variant: Any) -> str:
    variant = str(raw_variant or "foundation_v1").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "default": "foundation_v1",
        "foundation": "foundation_v1",
        "foundation_v1": "foundation_v1",
        "v1": "foundation_v1",
    }
    normalized = aliases.get(variant)
    if normalized is None:
        raise ValueError(
            f"Unsupported model.variant={raw_variant!r}. Only 'foundation_v1' is supported after cleanup."
        )
    return normalized


def _foundation_defaults() -> dict[str, Any]:
    return {
        "enc_channels": (32, 64, 128, 256, 512),
        "enc_down_strides": (2, 2, 2, 3),
        "enc_stage_num_res_blocks": (2, 2, 3, 3),
        "enc_kernel_size": 7,
        "latent_dim": 512,
        "quantizer_dim": 512,
        "codebook_size": 16384,
        "dec_channels": (512,),
        "decoder_dim": 512,
        "decoder_intermediate_dim": 1536,
        "decoder_num_layers": 12,
        "decoder_pos_net_enabled": True,
        "decoder_pos_net_dropout": 0.0,
        "decoder_pos_net_attention_heads": 12,
        "decoder_pos_net_attention_backend": "jax_cudnn",
        "istft_n_fft": 512,
        "istft_hop_length": 24,
        "istft_sample_rate": 5000.0,
        "istft_band_edges_hz": (200.0, 500.0, 1000.0),
        "istft_band_gain_init": (1.0, 0.8, 0.45, 0.12),
        "istft_dynamic_gate_scale": 0.5,
        "residual_correction_enabled": True,
        "residual_correction_alpha_init": 0.0,
        "residual_correction_alpha_max": 0.1,
        "residual_correction_hidden_dim": 128,
        "latent_bilstm_layers": 2,
        "latent_bilstm_hidden_dim": 256,
        "cnn_compute_dtype": "fp32",
        "param_dtype": "fp32",
        "diveq_sigma2": 1e-3,
        "search_chunk_size": 2048,
        "quant_conv_kernel_size": 7,
        "post_quant_conv_kernel_size": 7,
        "encoder_use_block_norm": True,
        "encoder_use_input_norm": True,
        "encoder_use_transition_norm": True,
    }


_REMOVED_MODEL_KEYS = frozenset(
    {
        "beta",
        "legacy_beta",
        "discriminator",
        "disc_dtype",
        "decoder_type",
        "dec_num_res_blocks",
        "dec_stage_num_res_blocks",
        "dec_up_strides",
        "dec_out_kernel_size",
        "decoder_use_block_norm",
        "decoder_use_upsample_norm",
        "decoder_stage_use_transformer",
        "decoder_stage_transformer_window_sizes",
        "decoder_stage_transformer_shift_sizes",
        "decoder_context_layers",
        "encoder_stage_use_transformer",
        "encoder_stage_transformer_window_sizes",
        "encoder_stage_transformer_shift_sizes",
        "pre_quant_transformer_layers",
        "post_quant_transformer_layers",
        "latent_transformer_type",
        "latent_transformer_window_size",
        "latent_transformer_shift_size",
        "stage_transformer_window_size",
        "stage_transformer_shift_size",
        "transformer_window_size",
        "transformer_shift_size",
        "transformer_heads",
        "transformer_mlp_ratio",
        "transformer_dropout",
        "transformer_ffn_activation",
        "transformer_attention_backend",
        "transformer_use_rope",
        "transformer_rope_base",
        "transformer_compute_dtype",
    }
)


def _validate_model_keys(model_cfg: dict[str, Any]) -> None:
    removed = sorted(key for key in model_cfg if key in _REMOVED_MODEL_KEYS)
    if removed:
        raise ValueError(
            "These model config keys belong to removed legacy modules and must be deleted: "
            f"{removed}"
        )
    supported = set(_foundation_defaults()) | {
        "variant",
        "enc_num_res_blocks",
        "compute_dtype",
        "residual_correction",
    }
    unknown = sorted(set(model_cfg) - supported)
    if unknown:
        raise ValueError(f"Unknown model config keys after cleanup: {unknown}")


def build_audio_model(model_cfg: dict[str, Any] | None) -> SimVQAudioModel:
    mcfg = dict(model_cfg or {})
    _validate_model_keys(mcfg)
    _normalize_variant(mcfg.get("variant", "foundation_v1"))
    merged_cfg = {**_foundation_defaults(), **mcfg}
    residual_cfg = dict(merged_cfg.get("residual_correction") or {})

    if int(merged_cfg.get("latent_bilstm_layers", 0)) <= 0:
        raise ValueError("foundation_v1 requires latent_bilstm_layers > 0.")

    cnn_dtype = _resolve_dtype(
        merged_cfg.get("cnn_compute_dtype", merged_cfg.get("compute_dtype", "fp32"))
    )
    param_dtype = _resolve_dtype(merged_cfg.get("param_dtype", "fp32"), fallback=jnp.float32)

    return SimVQAudioModel(
        in_channels=1,
        enc_channels=_tuple_cfg(merged_cfg, "enc_channels", (32, 64, 128, 256, 512)),
        enc_num_res_blocks=int(merged_cfg.get("enc_num_res_blocks", 4)),
        enc_stage_num_res_blocks=_optional_tuple_cfg(merged_cfg, "enc_stage_num_res_blocks"),
        enc_down_strides=_tuple_cfg(merged_cfg, "enc_down_strides", (2, 2, 2, 3)),
        latent_dim=int(merged_cfg.get("latent_dim", 512)),
        quantizer_dim=(
            None if merged_cfg.get("quantizer_dim") is None else int(merged_cfg.get("quantizer_dim"))
        ),
        codebook_size=int(merged_cfg.get("codebook_size", 16384)),
        dec_channels=_tuple_cfg(merged_cfg, "dec_channels", (512,)),
        enc_kernel_size=int(merged_cfg.get("enc_kernel_size", 7)),
        enc_dtype=cnn_dtype,
        dec_dtype=cnn_dtype,
        param_dtype=param_dtype,
        latent_bilstm_layers=int(merged_cfg.get("latent_bilstm_layers", 2)),
        latent_bilstm_hidden_dim=(
            None
            if merged_cfg.get("latent_bilstm_hidden_dim") is None
            else int(merged_cfg.get("latent_bilstm_hidden_dim"))
        ),
        diveq_sigma2=float(merged_cfg.get("diveq_sigma2", 1e-3)),
        search_chunk_size=int(merged_cfg.get("search_chunk_size", 2048)),
        quant_conv_kernel_size=int(merged_cfg.get("quant_conv_kernel_size", 7)),
        post_quant_conv_kernel_size=int(merged_cfg.get("post_quant_conv_kernel_size", 7)),
        encoder_use_block_norm=bool(merged_cfg.get("encoder_use_block_norm", True)),
        encoder_use_input_norm=bool(merged_cfg.get("encoder_use_input_norm", True)),
        encoder_use_transition_norm=bool(merged_cfg.get("encoder_use_transition_norm", True)),
        decoder_dim=int(merged_cfg.get("decoder_dim", 512)),
        decoder_intermediate_dim=int(merged_cfg.get("decoder_intermediate_dim", 1536)),
        decoder_num_layers=int(merged_cfg.get("decoder_num_layers", 12)),
        decoder_pos_net_enabled=bool(merged_cfg.get("decoder_pos_net_enabled", True)),
        decoder_pos_net_dropout=float(merged_cfg.get("decoder_pos_net_dropout", 0.0)),
        decoder_pos_net_attention_heads=int(merged_cfg.get("decoder_pos_net_attention_heads", 12)),
        decoder_pos_net_attention_backend=str(merged_cfg.get("decoder_pos_net_attention_backend", "jax_cudnn")),
        istft_n_fft=int(merged_cfg.get("istft_n_fft", 512)),
        istft_hop_length=int(merged_cfg.get("istft_hop_length", 24)),
        istft_sample_rate=float(merged_cfg.get("istft_sample_rate", 5000.0)),
        istft_band_edges_hz=tuple(float(v) for v in merged_cfg.get("istft_band_edges_hz", (200.0, 500.0, 1000.0))),
        istft_band_gain_init=tuple(float(v) for v in merged_cfg.get("istft_band_gain_init", (1.0, 0.8, 0.45, 0.12))),
        istft_dynamic_gate_scale=float(merged_cfg.get("istft_dynamic_gate_scale", 0.5)),
        residual_correction_enabled=bool(
            residual_cfg.get(
                "enabled",
                merged_cfg.get("residual_correction_enabled", True),
            )
        ),
        residual_correction_alpha_init=float(
            residual_cfg.get(
                "alpha_init",
                merged_cfg.get("residual_correction_alpha_init", 0.0),
            )
        ),
        residual_correction_alpha_max=float(
            residual_cfg.get(
                "alpha_max",
                merged_cfg.get("residual_correction_alpha_max", 0.1),
            )
        ),
        residual_correction_hidden_dim=int(
            residual_cfg.get(
                "hidden_dim",
                merged_cfg.get("residual_correction_hidden_dim", 128),
            )
        ),
    )


__all__ = ["build_audio_model"]
