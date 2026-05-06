from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from flax import linen as nn


class ReflectPadConv1d(nn.Module):
    features: int
    kernel: int
    stride: int = 1
    dilation: int = 1
    use_bias: bool = False
    feature_group_count: int = 1
    kernel_init: Any = nn.initializers.lecun_normal()
    bias_init: Any = nn.initializers.zeros
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    def setup(self) -> None:
        self.conv = nn.Conv(
            features=int(self.features),
            kernel_size=(int(self.kernel),),
            strides=(int(self.stride),),
            kernel_dilation=(int(self.dilation),),
            padding="VALID",
            use_bias=bool(self.use_bias),
            feature_group_count=int(self.feature_group_count),
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="conv",
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if x.ndim != 3:
            raise ValueError(f"ReflectPadConv1d expects (B,T,C), got {x.shape}")
        length = int(x.shape[1])
        kernel = int(self.kernel)
        stride = int(self.stride)
        dilation = int(self.dilation)
        effective_kernel = (kernel - 1) * dilation + 1
        out_length = (length + stride - 1) // stride
        total_pad = max(0, (out_length - 1) * stride + effective_kernel - length)
        pad_left = total_pad // 2
        pad_right = total_pad - pad_left
        if total_pad > 0:
            max_pad = max(pad_left, pad_right)
            if length <= max_pad:
                extra = max_pad - length + 1
                x = jnp.pad(x, ((0, 0), (0, extra), (0, 0)), mode="edge")
            x = jnp.pad(x, ((0, 0), (pad_left, pad_right), (0, 0)), mode="reflect")
            if length <= max_pad:
                x = x[:, : length + total_pad, :]
        return self.conv(x)


def Conv1d(
    features: int,
    kernel: int,
    stride: int = 1,
    dilation: int = 1,
    padding: str = "SAME",
    use_bias: bool = False,
    feature_group_count: int = 1,
    kernel_init: Any = nn.initializers.lecun_normal(),
    bias_init: Any = nn.initializers.zeros,
    dtype: Any = jnp.float32,
    param_dtype: Any = jnp.float32,
    name: str | None = None,
) -> nn.Module:
    if str(padding).strip().upper() in {"REFLECT", "REFLECT_SAME"}:
        return ReflectPadConv1d(
            features=features,
            kernel=kernel,
            stride=stride,
            dilation=dilation,
            use_bias=use_bias,
            feature_group_count=feature_group_count,
            kernel_init=kernel_init,
            bias_init=bias_init,
            dtype=dtype,
            param_dtype=param_dtype,
            name=name,
        )
    return nn.Conv(
        features=features,
        kernel_size=(kernel,),
        strides=(stride,),
        kernel_dilation=(dilation,),
        padding=padding,
        use_bias=use_bias,
        feature_group_count=feature_group_count,
        kernel_init=kernel_init,
        bias_init=bias_init,
        dtype=dtype,
        param_dtype=param_dtype,
        name=name,
    )
