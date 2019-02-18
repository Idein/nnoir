from .function import Function
from . import util


class MaxPooling2D(Function):
    def __init__(self, inputs, outputs, **params):
        required_params = {'kernel',
                           'stride',
                           'pad_h',
                           'pad_w'}
        optional_params = set()
        super(MaxPooling2D, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, x):
        img, col = util.im2col_cpu(x, self.params['kernel'], self.params['stride'],
                                   self.params['pad_h'], self.params['pad_w'],
                                   pval=-float('inf'))
        R = col.max(axis=(2, 3))
        return R
