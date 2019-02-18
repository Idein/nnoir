from .function import Function


class ReLU(Function):
    def __init__(self, inputs, outputs, **params):
        required_params = set()
        optional_params = set()
        super(ReLU, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, x):
        R = x.copy()
        R[R < 0] = 0
        return R
