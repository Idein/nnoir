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

    def run(self, x):
        kernel = (self.params['W'].shape[2], self.params['W'].shape[3])
        img, col = util.im2col_cpu(x, kernel, self.params['stride'],
                                   self.params['pad_h'],
                                   self.params['pad_w'], self.params['dilate'])
        R = np.tensordot(col, self.params['W'], ((1, 2, 3), (1, 2, 3))) + self.params['b']
        R = np.rollaxis(R, 3, 1)
        return R
