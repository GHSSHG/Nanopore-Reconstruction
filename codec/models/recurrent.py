from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn


class ResidualBiLSTMLayer1D(nn.Module):
    dim: int
    hidden_dim: int
    forget_bias: float = 1.0
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    def setup(self) -> None:
        gate_dim = 4 * int(self.hidden_dim)
        kernel_init = nn.initializers.xavier_uniform()
        self.fwd_x_kernel = self.param(
            "fwd_x_kernel",
            kernel_init,
            (int(self.dim), gate_dim),
            self.param_dtype,
        )
        self.fwd_x_bias = self.param(
            "fwd_x_bias",
            nn.initializers.zeros,
            (gate_dim,),
            self.param_dtype,
        )
        self.fwd_h_kernel = self.param(
            "fwd_h_kernel",
            kernel_init,
            (int(self.hidden_dim), gate_dim),
            self.param_dtype,
        )
        self.bwd_x_kernel = self.param(
            "bwd_x_kernel",
            kernel_init,
            (int(self.dim), gate_dim),
            self.param_dtype,
        )
        self.bwd_x_bias = self.param(
            "bwd_x_bias",
            nn.initializers.zeros,
            (gate_dim,),
            self.param_dtype,
        )
        self.bwd_h_kernel = self.param(
            "bwd_h_kernel",
            kernel_init,
            (int(self.hidden_dim), gate_dim),
            self.param_dtype,
        )
        self.out_proj = nn.Dense(
            int(self.dim),
            use_bias=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="out_proj",
        )

    def _run_direction(
        self,
        x: jnp.ndarray,
        *,
        x_kernel: jnp.ndarray,
        x_bias: jnp.ndarray,
        h_kernel: jnp.ndarray,
    ) -> jnp.ndarray:
        batch = int(x.shape[0])
        hidden = int(self.hidden_dim)
        x_time = jnp.swapaxes(x, 0, 1)
        h0 = jnp.zeros((batch, hidden), dtype=self.dtype)
        c0 = jnp.zeros((batch, hidden), dtype=self.dtype)
        x_gates = jnp.matmul(x_time, x_kernel.astype(self.dtype)) + x_bias.astype(self.dtype)
        forget_bias = jnp.asarray(float(self.forget_bias), dtype=self.dtype)

        def _step(carry: tuple[jnp.ndarray, jnp.ndarray], gates_x: jnp.ndarray):
            h_prev, c_prev = carry
            gates = gates_x + jnp.matmul(h_prev, h_kernel.astype(self.dtype))
            i, f, g, o = jnp.split(gates, 4, axis=-1)
            i = nn.sigmoid(i)
            f = nn.sigmoid(f + forget_bias)
            g = jnp.tanh(g)
            o = nn.sigmoid(o)
            c = f * c_prev + i * g
            h = o * jnp.tanh(c)
            return (h, c), h

        (_, _), y_time = jax.lax.scan(_step, (h0, c0), x_gates)
        return jnp.swapaxes(y_time, 0, 1)

    def __call__(self, x: jnp.ndarray, *, train: bool = False) -> jnp.ndarray:
        del train
        if x.ndim != 3:
            raise ValueError(f"ResidualBiLSTMLayer1D expects (B,T,C), got {x.shape}")
        if int(x.shape[-1]) != int(self.dim):
            raise ValueError(f"Input dim {x.shape[-1]} != layer dim {self.dim}")
        h = x.astype(self.dtype)
        fwd = self._run_direction(
            h,
            x_kernel=self.fwd_x_kernel,
            x_bias=self.fwd_x_bias,
            h_kernel=self.fwd_h_kernel,
        )
        bwd_in = jnp.flip(h, axis=1)
        bwd = self._run_direction(
            bwd_in,
            x_kernel=self.bwd_x_kernel,
            x_bias=self.bwd_x_bias,
            h_kernel=self.bwd_h_kernel,
        )
        bwd = jnp.flip(bwd, axis=1)
        y = self.out_proj(jnp.concatenate((fwd, bwd), axis=-1))
        return (h + y).astype(self.dtype)


class ResidualBiLSTM1D(nn.Module):
    dim: int
    num_layers: int = 2
    hidden_dim: int | None = None
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    def setup(self) -> None:
        hidden = self.hidden_dim
        if hidden is None:
            hidden = max(1, int(self.dim) // 2)
        if int(self.num_layers) <= 0:
            raise ValueError("ResidualBiLSTM1D requires num_layers > 0.")
        self.layers = tuple(
            ResidualBiLSTMLayer1D(
                dim=int(self.dim),
                hidden_dim=int(hidden),
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name=f"layer_{idx}",
            )
            for idx in range(int(self.num_layers))
        )

    def __call__(self, x: jnp.ndarray, *, train: bool = False) -> jnp.ndarray:
        h = x
        for layer in self.layers:
            h = layer(h, train=train)
        return h
