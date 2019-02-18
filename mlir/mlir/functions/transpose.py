from .function import Function


class Transpose(Function):
    def __init__(self, inputs, outputs, **params):
        required_params = {'axes'}
        optional_params = set()
        super(Transpose, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, x):
        return x.transpose(self.params['axes'])
