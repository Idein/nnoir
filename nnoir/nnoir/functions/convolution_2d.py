from .function import Function
import numpy as np
from . import util


class Convolution2D(Function):
    def __init__(self, inputs, outputs, **params):
        required_params = {'W',
                           'b',
                           'pad_h',
                           'pad_w',
                           'stride',
                           'dilate',
                           'groups'}
        optional_params = set()
        super(Convolution2D, self).__init__(inputs, outputs, params, required_params, optional_params)

    def grouped_convolution(self, x):
        G = self.params['groups']
        batch, in_ch, in_h, in_w = x.shape
        out_ch, _, kh, kw = self.params['W'].shape
        icg = in_ch // G
        ocg = out_ch // G

        kernel = (self.params['W'].shape[2], self.params['W'].shape[3])
        _, x = util.im2col_cpu(x, kernel, self.params['stride'],
                               self.params['pad_h'],
                               self.params['pad_w'], self.params['dilate'])
        out_h, out_w = x.shape[-2:]

        x = x.transpose(1, 2, 3, 0, 4, 5)  # (in_ch, kh, kw, batch, out_h, out_w)
        x = x.reshape(G, icg * kh * kw, batch * out_h * out_w)

        W = self.params['W'].reshape(G, ocg, icg * kh * kw)

        y = np.matmul(W, x).astype(x.dtype, copy=False)
        y = y.reshape(out_ch, batch, out_h, out_w)
        y = y.transpose(1, 0, 2, 3)
        if self.params['b'] is not None:
            y += self.params['b'].reshape(1, self.params['b'].size, 1, 1)

        return y,

    def run(self, x):
        if self.params['groups'] > 1:
            return self.grouped_convolution(x)

        kernel = (self.params['W'].shape[2], self.params['W'].shape[3])
        img, col = util.im2col_cpu(x, kernel, self.params['stride'],
                                   self.params['pad_h'],
                                   self.params['pad_w'], self.params['dilate'])
        R = np.tensordot(col, self.params['W'], ((1, 2, 3), (1, 2, 3)))
        if self.params['b'] is not None:
            R += self.params['b']
        R = np.rollaxis(R, 3, 1)
        return R
