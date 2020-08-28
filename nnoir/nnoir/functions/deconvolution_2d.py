import numpy as np

from .function import Function
from . import util


class Deconvolution2D(Function):
    def __init__(self, inputs, outputs, **params):
        required_params = {'W',
                           'b',
                           'pad_h',
                           'pad_w',
                           'stride',
                           'dilate',
                           'groups'}
        optional_params = set()
        super(Deconvolution2D, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, x):
        W = self.params['W']
        b = self.params['b']
        ph_pre, ph_post = self.params['pad_h']
        pw_pre, pw_post = self.params['pad_w']

        _, _, in_h, in_w = x.shape
        _, _, kh, kw = W.shape
        sy = self.params['stride'][0]
        sx = self.params['stride'][1]
        dy = self.params['dilate'][0]
        dx = self.params['dilate'][1]

        outh = util.get_deconv_outsize(in_h, kh, sx, dx, self.params['pad_h'], (0, 0))
        outw = util.get_deconv_outsize(in_w, kw, sy, dy, self.params['pad_w'], (0, 0))

        gcol = np.tensordot(W, x, (0, 1)).astype(x.dtype, copy=False)
        gcol = np.rollaxis(gcol, 3)

        y = util.col2im_cpu(gcol, self.params['stride'], ph_pre, pw_pre,
                            outh, outw, dy=dy, dx=dx)  # ph_post, pw_postは正しい？
        # b, k, h, w
        if b is not None:
            y += b.reshape((1, b.size, 1, 1))
        return y
