import numpy as np

from . import util
from .function import Function


class Linear(Function):
    def __init__(self, inputs, outputs, **params):
        required_params = {"W", "b"}  # (out_ch, in_ch)
        optional_params = {"y_scale", "y_zero", "w_scale", "w_zero"}
        super(Linear, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, x):
        x_ = x.reshape(x.shape[0], -1)
        w_ = np.transpose(self.params["W"])
        if self.params["W"].dtype == np.uint8:
            func = lambda x, w: x.dot(w)
            result = util.calc_with_uint8_weight(func, x_, w_, self.params["w_scale"], self.params["w_zero"])
        else:
            assert self.params["W"].dtype == np.float32
            result = x_.dot(w_)
        if self.params["b"] is not None:
            assert self.params["b"].dtype == np.float32
            result += self.params["b"]
        return result
