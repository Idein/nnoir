from .function import Function


class Reshape(Function):
    def __init__(self, inputs, outputs, **params):
        required_params = {'shape'}
        optional_params = set()
        super(Reshape, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, x):
        return x.reshape(self.params['shape'])
