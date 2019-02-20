import numpy as np
from .function import Function
from . import util


class DepthwiseConvolution2D(Function):
    def __init__(self, inputs, outputs, **params):
        required_params = {'W',
                           'b',
                           'stride',
                           'pad_h',
                           'pad_w',
                           'dilate'}
        optional_params = set()
        super(DepthwiseConvolution2D, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, x):
        W = self.params['W']
        b = self.params['b']
        kernel = (self.params['W'].shape[2], self.params['W'].shape[3])
        img, col = util.im2col_cpu(x, kernel, self.params['stride'],
                                   self.params['pad_h'],
                                   self.params['pad_w'],
                                   self.params['dilate'])
        B, C, KY, KX, IY, IX = col.shape
        D = W.shape[0]  # (D, C, KY, KX)
        c_ = col.transpose(1, 0, 4, 5, 2, 3) .reshape((C, B * IY * IX, KY * KX))
        w_ = W.transpose(1, 2, 3, 0).reshape((C, KY * KX, D))
        y = np.matmul(c_, w_).astype(col.dtype, copy=False)
        y = y.reshape((C, B, IY * IX, D)).transpose(1, 0, 3, 2).reshape((B, C * D, IY, IX))
        if b is not None:
            y += b[None, :, None, None]
        return y
