import chainer.functions as F


class ConvertTranspose():

    def __init__(self, function, inputs, outputs):
        self.f = lambda x: F.transpose(x, function.params['axes'])

    def __call__(self, x):
        return self.f(x)
