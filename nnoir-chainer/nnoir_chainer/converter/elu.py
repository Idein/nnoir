import chainer.functions as F


class ConvertELU():

    def __init__(self, function, inputs, outputs):
        self.f = lambda x: F.elu(x, function.params['alpha'])

    def __call__(self, x):
        return self.f(x)
