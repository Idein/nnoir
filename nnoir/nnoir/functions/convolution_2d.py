from typing import Any, List, Set

import numpy as np

from . import util
from .function import Function


class Convolution2D(Function):
    def __init__(self, inputs: List[bytes], outputs: List[bytes], **params: Any):
        required_params = {"W", "b", "pad_h", "pad_w", "stride", "dilate", "groups"}
        optional_params = {"y_scale", "y_zero", "w_scale", "w_zero"}
        super(Convolution2D, self).__init__(inputs, outputs, params, required_params, optional_params)

    def grouped_convolution(self, x):  # type: ignore
        G = self.params["groups"]
        batch, in_ch, in_h, in_w = x.shape
        out_ch, _, kh, kw = self.params["W"].shape
        icg = in_ch // G
        ocg = out_ch // G

        kernel = (self.params["W"].shape[2], self.params["W"].shape[3])
        _, x = util.im2col_cpu(
            x,
            kernel,
            self.params["stride"],
            self.params["pad_h"],
            self.params["pad_w"],
            self.params["dilate"],
        )
        out_h, out_w = x.shape[-2:]

        x = x.transpose(1, 2, 3, 0, 4, 5)  # (in_ch, kh, kw, batch, out_h, out_w)
        x = x.reshape(G, icg * kh * kw, batch * out_h * out_w)

        W = self.params["W"].reshape(G, ocg, icg * kh * kw)

        if self.params["W"].dtype == np.uint8:
            func = lambda x, w: np.matmul(w, x).astype(x.dtype, copy=False)
            y = util.calc_with_uint8_weight(func, x, W, self.params["w_scale"], self.params["w_zero"])  # type: ignore
        else:
            assert W.dtype == np.float32
            y = np.matmul(W, x).astype(x.dtype, copy=False)
        y = y.reshape(out_ch, batch, out_h, out_w)
        y = y.transpose(1, 0, 2, 3)
        if self.params["b"] is not None:
            assert self.params["b"].dtype == np.float32
            y += self.params["b"].reshape(1, self.params["b"].size, 1, 1)

        return (y,)

    def run(self, x):  # type: ignore
        if self.params["groups"] > 1:
            return self.grouped_convolution(x)  # type: ignore

        kernel = (self.params["W"].shape[2], self.params["W"].shape[3])
        img, col = util.im2col_cpu(
            x,
            kernel,
            self.params["stride"],
            self.params["pad_h"],
            self.params["pad_w"],
            self.params["dilate"],
        )
        if self.params["W"].dtype == np.uint8:
            func = lambda x, w: np.tensordot(x, w, ((1, 2, 3), (1, 2, 3)))
            R = util.calc_with_uint8_weight(func, col, self.params["W"], self.params["w_scale"], self.params["w_zero"])  # type: ignore
        else:
            assert self.params["W"].dtype == np.float32
            R = np.tensordot(col, self.params["W"], ((1, 2, 3), (1, 2, 3)))
        if self.params["b"] is not None:
            assert self.params["b"].dtype == np.float32
            R += self.params["b"]
        R = np.rollaxis(R, 3, 1)
        return R
