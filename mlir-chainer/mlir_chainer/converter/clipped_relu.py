import chainer.functions as F


class ConvertClippedReLU():

    def __init__(self, function, inputs, outputs):
        self.f = lambda x: F.clipped_relu(x, function.params['upper'])

    def __call__(self, x):
        return self.f(x)
