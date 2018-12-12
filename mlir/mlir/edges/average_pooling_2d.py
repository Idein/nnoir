from .edge import Edge
import chainer.functions as F
from . import util

class AveragePooling2D(Edge):
    def __init__(self, inputs, outputs, **params):
        necessary_params = {'kernel',
                            'stride',
                            'pad_h',
                            'pad_w'}
        optional_params = set()
        super(AveragePooling2D, self).__init__(inputs, outputs, params, necessary_params, optional_params)
    def run(self, x):
        img, col = util.im2col_cpu(x, self.params['kernel'], self.params['stride'],
                                   self.params['pad_h'], self.params['pad_w'])
        return col.mean(axis=(2,3))
