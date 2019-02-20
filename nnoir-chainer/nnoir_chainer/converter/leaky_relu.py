import chainer.functions as F


class ConvertLeakyReLU():

    def __init__(self, function, inputs, outputs):
        self.f = lambda x: F.leaky_relu(x, function.params['slope'])

    def __call__(self, x):
        return self.f(x)
