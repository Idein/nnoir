import numpy as np

from . import util
from .function import Function


class DepthwiseConvolution2D(Function):
    def __init__(self, inputs, outputs, **params):
        required_params = {"W", "b", "stride", "pad_h", "pad_w", "dilate"}
        optional_params = {"y_scale", "y_zero", "w_scale", "w_zero"}
        super(DepthwiseConvolution2D, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, x):
        W = self.params["W"]
        b = self.params["b"]
        kernel = (self.params["W"].shape[2], self.params["W"].shape[3])
        img, col = util.im2col_cpu(
            x,
            kernel,
            self.params["stride"],
            self.params["pad_h"],
            self.params["pad_w"],
            self.params["dilate"],
        )
        B, C, KY, KX, IY, IX = col.shape
        D = W.shape[0]  # (D, C, KY, KX)
        c_ = col.transpose(1, 0, 4, 5, 2, 3).reshape((C, B * IY * IX, KY * KX))
        w_ = W.transpose(1, 2, 3, 0).reshape((C, KY * KX, D))
        if self.params["W"].dtype == np.uint8:
            func = lambda x, w: np.matmul(x, w).astype(col.dtype, copy=False)
            y = util.calc_with_uint8_weight(func, c_, w_, self.params["w_scale"], self.params["w_zero"])
        else:
            assert self.params["W"].dtype == np.float32
            y = np.matmul(c_, w_).astype(col.dtype, copy=False)
        y = y.reshape((C, B, IY * IX, D)).transpose(1, 0, 3, 2).reshape((B, C * D, IY, IX))
        if b is not None:
            assert self.params["W"].dtype == np.float32
            y += b[None, :, None, None]
        return y
