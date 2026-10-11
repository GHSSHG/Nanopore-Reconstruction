from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn

LSTM_UNROLL = 8  # time steps per loop iteration of the recurrence


def _lstm_cell(state, gates_x: jnp.ndarray, h_kernel: jnp.ndarray, forget_bias: jnp.ndarray):
    h_prev, c_prev = state
    gates = gates_x + jnp.matmul(h_prev, h_kernel)
    i, f, g, o = jnp.split(gates, 4, axis=-1)
    i = nn.sigmoid(i)
    f = nn.sigmoid(f + forget_bias)
    g = jnp.tanh(g)
    o = nn.sigmoid(o)
    c = f * c_prev + i * g
    h = o * jnp.tanh(c)
    return h, c


def _bilstm(x: jnp.ndarray, fwd_params, bwd_params, *, hidden: int, forget_bias: float, dtype: Any):
    """Both directions of a layer in one scan: step t runs the forward cell on x[:, t] and the
    backward cell on x[:, T-1-t]. Each cell computes exactly what a scan per direction would;
    sharing the loop halves the sequential iterations, which bound the time on a GPU."""

    def input_gates(seq, x_kernel, x_bias):
        return jnp.matmul(jnp.swapaxes(seq, 0, 1), x_kernel.astype(dtype)) + x_bias.astype(dtype)

    gates_f = input_gates(x, fwd_params[0], fwd_params[1])
    gates_b = input_gates(jnp.flip(x, axis=1), bwd_params[0], bwd_params[1])
    h_kernel_f, h_kernel_b = fwd_params[2].astype(dtype), bwd_params[2].astype(dtype)
    fb = jnp.asarray(forget_bias, dtype=dtype)
    zeros = jnp.zeros((x.shape[0], hidden), dtype=dtype)

    def step(carry, gates):
        fwd = _lstm_cell(carry[0], gates[0], h_kernel_f, fb)
        bwd = _lstm_cell(carry[1], gates[1], h_kernel_b, fb)
        return (fwd, bwd), (fwd[0], bwd[0])

    _, (y_f, y_b) = jax.lax.scan(step, ((zeros, zeros), (zeros, zeros)), (gates_f, gates_b), unroll=LSTM_UNROLL)
    return jnp.swapaxes(y_f, 0, 1), jnp.flip(jnp.swapaxes(y_b, 0, 1), axis=1)


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

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if x.ndim != 3:
            raise ValueError(f"ResidualBiLSTMLayer1D expects (B,T,C), got {x.shape}")
        if int(x.shape[-1]) != int(self.dim):
            raise ValueError(f"Input dim {x.shape[-1]} != layer dim {self.dim}")
        h = x.astype(self.dtype)
        fwd, bwd = _bilstm(
            h,
            (self.fwd_x_kernel, self.fwd_x_bias, self.fwd_h_kernel),
            (self.bwd_x_kernel, self.bwd_x_bias, self.bwd_h_kernel),
            hidden=int(self.hidden_dim),
            forget_bias=float(self.forget_bias),
            dtype=self.dtype,
        )
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

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = x
        for layer in self.layers:
            h = layer(h)
        return h
