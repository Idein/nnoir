import chainer.functions as F


class ConvertBilinear2D():

    def __init__(self, function, inputs, outputs):
        self.f = lambda x: F.resize_images(x, tuple(function.params['size']))

    def __call__(self, x):
        return self.f(x)
