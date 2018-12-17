import chainer.functions as F

class ConvertConstantPadding():

    def __init__(self, edge, inputs, outputs):
        pad_width = list(map(tuple, edge.params['pads']))
        self.f = lambda x: F.pad(x, pad_width, 'constant', constant_values=[edge.params['value']])

    def __call__(self, x):
        return self.f(x)
