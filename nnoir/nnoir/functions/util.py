from typing import Any, Callable, List, Set, Tuple

import numpy as np
from numpy.typing import NDArray


def get_conv_outsize(size: int, k: int, s: int, pre_p: int, post_p: int, cover_all: bool = False, d: int = 1) -> int:
    dk = k + (k - 1) * (d - 1)
    return (size + pre_p + post_p - dk) // s + 1


def im2col_cpu(
    img: NDArray[Any],
    kernel: Tuple[int, int],
    stride: Tuple[int, int],
    ph: Tuple[int, int],
    pw: Tuple[int, int],
    dilate: Tuple[int, int] = (1, 1),
    pval: float = 0,
    cover_all: bool = False,
) -> Tuple[NDArray[Any], NDArray[Any]]:
    n, c, h, w = img.shape
    kh, kw = kernel
    sy, sx = stride
    dy, dx = dilate
    pre_ph = ph[0]
    post_ph = ph[1]
    pre_pw = pw[0]
    post_pw = pw[1]
    out_h = get_conv_outsize(h, kh, sy, pre_ph, post_ph, cover_all, dy)
    out_w = get_conv_outsize(w, kw, sx, pre_pw, post_pw, cover_all, dx)

    img = np.pad(  # type: ignore
        img,
        ((0, 0), (0, 0), (pre_ph, post_ph), (pre_pw, post_pw)),
        mode="constant",
        constant_values=(pval,),
    )
    col = np.ndarray((n, c, kh, kw, out_h, out_w), dtype=img.dtype)  # type: ignore

    for j in range(kh):
        jdy = j * dy
        j_lim = jdy + sy * out_h
        for i in range(kw):
            idx = i * dx
            i_lim = idx + sx * out_w
            col[:, :, j, i, :, :] = img[:, :, jdy:j_lim:sy, idx:i_lim:sx]

    return img, col


def col2im_cpu(
    col: NDArray[Any], stride: Tuple[int, int], ph: int, pw: int, h: int, w: int, dy: int = 1, dx: int = 1
) -> NDArray[Any]:
    sy, sx = stride
    n, c, kh, kw, out_h, out_w = col.shape
    img = np.zeros((n, c, h + 2 * ph + sy - 1, w + 2 * pw + sx - 1), dtype=col.dtype)
    for j in range(kh):
        jdy = j * dy
        j_lim = jdy + sy * out_h
        for i in range(kw):
            idx = i * dx
            i_lim = idx + sx * out_w
            img[:, :, jdy:j_lim:sy, idx:i_lim:sx] += col[:, :, j, i]
    return img[:, :, ph : h + ph, pw : w + pw]  # type: ignore


def get_quantization_factors(_min, _max):  # type: ignore
    _max = max(_max, 0)
    _min = min(_min, 0)
    scale = ((_max - _min) / 255).astype("float32")
    zero = -_min / scale
    zero = np.round(zero.clip(0, 255)).astype("uint8")  # type: ignore
    return np.asscalar(scale), np.asscalar(zero)  # type: ignore


def quantize_variable(x):  # type: ignore
    scale, zero = get_quantization_factors(x.astype("float32").min(), x.astype("float32").max())  # type: ignore
    quantized_x = np.round((x / scale + zero).clip(0, 255)).astype("uint8")  # type: ignore
    return quantized_x, scale, zero


def calc_with_uint8_weight(func, x_fp32, w_uint8, w_scale, w_zero):  # type: ignore
    x_uint8, x_scale, x_zero = quantize_variable(x_fp32)  # type: ignore
    x = x_uint8.astype("int32") - int(x_zero)
    w = w_uint8.astype("int32") - int(w_zero)
    y = func(x, w).astype("float32") * x_scale * w_scale
    return y
