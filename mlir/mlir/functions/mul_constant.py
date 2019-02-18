from .function import Function


class MulConstant(Function):
    def __init__(self, inputs, outputs, **params):
        required_params = {'value'}
        optional_params = set()
        super(MulConstant, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, x):
        return x * self.params['value']
