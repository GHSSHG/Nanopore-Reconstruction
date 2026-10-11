from __future__ import annotations

from typing import Any, Dict, NamedTuple, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pl_triton

F32 = jnp.float32
F16 = jnp.float16


def pow2_scale(bound: jnp.ndarray) -> jnp.ndarray:
    """2^k with bound * 2^k < 2^15: fits fp16 (max 65504) with headroom; applying and undoing it
    is exact."""
    _, exp = jnp.frexp(jnp.maximum(jnp.asarray(bound, F32), jnp.finfo(F32).tiny))
    return jnp.ldexp(F32(1.0), jnp.clip(15 - exp, -100, 100))


def to_f16(x: jnp.ndarray) -> jnp.ndarray:
    """fp32 -> fp16 rounded to nearest with ties away from zero, as Ampere GPUs convert fp32 to
    TF32 (measured for cuBLAS and cuDNN); exact for fp16's normal range."""
    bits = jax.lax.bitcast_convert_type(x.astype(F32), jnp.uint32)
    bits = (bits + jnp.uint32(0x1000)) & jnp.uint32(0xFFFFE000)
    return jax.lax.bitcast_convert_type(bits, F32).astype(F16)


# Codebook search tile: latent rows, codewords, dimensions per step, then Triton warps and
# pipeline stages. Same as the NanoRecon app; changing it never changes the codes.
SEARCH_TILE = (64, 256, 32, 4, 3)


class SearchCodebook(NamedTuple):
    """The projected codebook prepared for the search."""

    codes: jnp.ndarray  # fp16 (K, D): codebook * 2^k, zero-padded to whole search tiles
    inverse_scale: jnp.ndarray  # 2^-k
    norms: jnp.ndarray  # fp32 (K,): ||e||^2 of the fp32 codebook, inf for padding rows


def search_codebook(codebook: jnp.ndarray) -> SearchCodebook:
    _, tile_codes, tile_dims, _, _ = SEARCH_TILE
    num_codes, dim = codebook.shape
    scale = pow2_scale(jnp.max(jnp.abs(codebook)))
    codes = jnp.pad(to_f16(codebook * scale), ((0, -num_codes % tile_codes), (0, -dim % tile_dims)))
    norms = jnp.pad(jnp.sum(codebook.astype(F32) ** 2, axis=1), (0, -num_codes % tile_codes), constant_values=jnp.inf)
    return SearchCodebook(codes, 1.0 / scale, norms)


def nearest_codeword(z: jnp.ndarray, codebook: SearchCodebook) -> jnp.ndarray:
    """Exact nearest neighbour (squared L2) of each row of z (N, D) in the codebook.

    Distances ||z||^2 + ||e||^2 - 2 z.e with z.e on fp16 operands (TF32-equal inputs, fp32
    sums). One GPU kernel computes the distances tile by tile and keeps only each row's minimum
    per tile with its first index, so the N x K distances never reach GPU memory; the tile minima
    are then compared in codebook order, so ties resolve to the lowest index. Same kernel as the
    NanoRecon app, so training and the app pick the same codes.
    """
    rows, codes, dims, warps, stages = SEARCH_TILE
    num_rows = z.shape[0]
    num_codes, dim = codebook.codes.shape
    z = z.astype(F32)
    z_scale = pow2_scale(jnp.max(jnp.abs(z), axis=1))  # per row: rows never affect each other
    pad = (0, -num_rows % rows)
    z16 = jnp.pad(to_f16(z * z_scale[:, None]), (pad, (0, dim - z.shape[1])))
    unscale = jnp.pad(codebook.inverse_scale / z_scale, pad)
    z_norm = jnp.pad(jnp.sum(z**2, axis=1), pad)

    def kernel(z_ref, codes_ref, unscale_ref, z_norm_ref, norms_ref, low_ref, first_ref):
        def step(k, acc):
            cols = pl.ds(k * dims, dims)
            return acc + pl.dot(z_ref[:, cols], codes_ref[:, cols], trans_b=True)

        dots = jax.lax.fori_loop(0, dim // dims, step, jnp.zeros((rows, codes), F32))
        dists = z_norm_ref[...][:, None] + norms_ref[...][None, :] - 2.0 * (dots * unscale_ref[...][:, None])
        low = jnp.min(dists, axis=1)
        first = jnp.min(jnp.where(dists == low[:, None], jnp.arange(codes, dtype=jnp.int32)[None, :], codes), axis=1)
        low_ref[0, :] = low
        first_ref[0, :] = first + pl.program_id(1) * codes

    tiles = (z16.shape[0] // rows, num_codes // codes)
    out = jax.ShapeDtypeStruct((tiles[1], z16.shape[0]), F32)
    lows, firsts = pl.pallas_call(
        kernel,
        out_shape=(out, out.update(dtype=jnp.int32)),
        grid=tiles,
        in_specs=(
            pl.BlockSpec((rows, dim), lambda i, j: (i, 0)),
            pl.BlockSpec((codes, dim), lambda i, j: (j, 0)),
            pl.BlockSpec((rows,), lambda i, j: (i,)),
            pl.BlockSpec((rows,), lambda i, j: (i,)),
            pl.BlockSpec((codes,), lambda i, j: (j,)),
        ),
        out_specs=(pl.BlockSpec((1, rows), lambda i, j: (j, i)),) * 2,
        compiler_params=pl_triton.CompilerParams(num_warps=warps, num_stages=stages),
        interpret=jax.default_backend() == "cpu",
        name="nearest_codeword",
    )(z16, codebook.codes, unscale, z_norm, codebook.norms)
    best_tile = jnp.argmin(lows, axis=0)  # the first tile holding the minimum
    return jnp.take_along_axis(firsts, best_tile[None, :], axis=0)[0, :num_rows]


class SimVQ1D(nn.Module):
    """SimVQ quantizer for 1D latents (B, T, C) with deterministic DiVeQ gradients.

    The forward value is always the nearest projected codeword, so training and inference
    compute the same thing. Gradients follow DiVeQ without noise: z_q acts as
    z + ||q - z|| * stop_grad((q - z) / ||q - z||), which moves z and the codebook projection
    along the quantization error.
    """

    codebook_size: int
    code_dim: int

    def setup(self) -> None:
        embed_init = nn.initializers.normal(stddev=self.code_dim**-0.5)
        # SimVQ keeps the base embeddings fixed and trains projection params to adapt latents.
        self.codebook_var = self.variable(
            "vq",
            "codebook",
            lambda: embed_init(self.make_rng("params"), (self.codebook_size, self.code_dim)).astype(F32),
        )
        self.W = self.param("W", nn.initializers.orthogonal(), (self.code_dim, self.code_dim), F32)
        self.proj_bias = self.param("proj_bias", nn.initializers.zeros, (self.code_dim,), F32)

    def projected_codebook(self) -> jnp.ndarray:
        emb = jax.lax.stop_gradient(self.codebook_var.value).astype(F32)
        return jnp.dot(emb, self.W) + self.proj_bias

    def __call__(self, z: jnp.ndarray) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        B, T, C = z.shape
        if C != self.code_dim:
            raise ValueError(f"Latent dim {C} != code_dim {self.code_dim}")
        codebook = self.projected_codebook()  # (K, D)
        search = search_codebook(jax.lax.stop_gradient(codebook))
        indices = nearest_codeword(jax.lax.stop_gradient(z).reshape(B * T, C), search)
        quant = jnp.take(codebook, indices, axis=0).reshape(B, T, C)

        diff = quant - z
        diff_norm = jnp.sqrt(jnp.sum(jnp.square(diff), axis=-1, keepdims=True) + 1e-16)
        diveq = z + diff_norm * jax.lax.stop_gradient(diff / diff_norm)
        # Forward value is exactly the codeword; the gradient is that of the DiVeQ expression.
        z_q = jax.lax.stop_gradient(quant) + (diveq - jax.lax.stop_gradient(diveq))

        info = {
            "indices": indices.reshape(B, T),
            "q_z_dist": jnp.mean(diff_norm),
            "log_q_z_dist": jnp.mean(jnp.log(diff_norm)),
        }
        return z_q, info
