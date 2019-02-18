from .function import Function


class SoftmaxCrossEntropy(Function):
    def __init__(self, inputs, outputs, **params):
        required_params = {'normalize',
                           'cache_score'}
        optional_params = set()
        super(SoftmaxCrossEntropy, self).__init__(inputs, outputs, params, required_params, optional_params)
