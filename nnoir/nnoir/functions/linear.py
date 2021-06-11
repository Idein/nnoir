import numpy as np

from .function import Function


class Linear(Function):
    def __init__(self, inputs, outputs, **params):
        required_params = {"W", "b"}  # (out_ch, in_ch)
        optional_params = {"y_scale", "y_zero", "w_scale", "w_zero"}
        super(Linear, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, x):
        result = x.reshape(x.shape[0], -1).dot(np.transpose(self.params["W"]))
        if self.params["b"] is not None:
            result += self.params["b"]
        return result
