import chainer.functions as F


class ConvertConstantPadding():

    def __init__(self, function, inputs, outputs):
        pad_width = list(map(tuple, function.params['pads']))
        self.f = lambda x: F.pad(x, pad_width, 'constant', constant_values=[function.params['value']])

    def __call__(self, x):
        return self.f(x)
