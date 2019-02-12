import chainer.functions as F
from .function import Function
from . import util
import numpy as np

class Convolution2DFunction(Function):
    def __init__(self, inputs, outputs, **params):
        required_params = {'stride', 'pad_h', 'pad_w', 'dilate', 'groups'}
        optional_params = set()
        super(Convolution2DFunction, self).__init__(inputs, outputs, params, required_params, optional_params)
    def run(self, x, W, b):
        kernel = (W.shape[2], W.shape[3])
        img, col = util.im2col_cpu(x, kernel, self.params['stride'],
                                   self.params['pad_h'],
                                   self.params['pad_w'], self.params['dilate'])
        R = np.tensordot(col, W, ((1, 2, 3), (1, 2, 3))) + b
        R = np.rollaxis(R, 3, 1)
        return R
