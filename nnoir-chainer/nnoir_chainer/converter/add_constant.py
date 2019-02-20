import chainer.functions as F


class ConvertAddConstant():

    def __init__(self, function, inputs, outputs):
        self.f = lambda x: x + function.params['value']

    def __call__(self, x):
        return self.f(x)
