from .function import Function
from . import util


class Constant(Function):
    def __init__(self, inputs, outputs, **params):
        required_params = {'value'}
        optional_params = set()
        super(Constant, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self):
        return self.params['value']
