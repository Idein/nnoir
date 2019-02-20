import chainer.functions as F


class ConvertSoftmax():

    def __init__(self, function, inputs, outputs):
        self.f = lambda x: F.softmax(x, function.params['axis'])

    def __call__(self, x):
        return self.f(x)
