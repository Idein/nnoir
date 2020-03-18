import numpy as np

from .function import Function
from . import util


class AveragePooling2D(Function):
    def __init__(self, inputs, outputs, **params):
        required_params = {'kernel',
                           'stride',
                           'pad_h',
                           'pad_w',
                           'count_exclude_pad'}
        optional_params = set()
        super(AveragePooling2D, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, x):
        if self.params['count_exclude_pad']:
            img, col = util.im2col_cpu(x, self.params['kernel'], self.params['stride'],
                                       self.params['pad_h'], self.params['pad_w'])
            _, mask = util.im2col_cpu(np.ones(x.shape, dtype=np.int32), self.params['kernel'], self.params['stride'],
                                      self.params['pad_h'], self.params['pad_w'])
            return col.sum(axis=(2, 3)) / mask.sum(axis=(2, 3))
        else:
            img, col = util.im2col_cpu(x, self.params['kernel'], self.params['stride'],
                                       self.params['pad_h'], self.params['pad_w'])
            return col.mean(axis=(2, 3))
