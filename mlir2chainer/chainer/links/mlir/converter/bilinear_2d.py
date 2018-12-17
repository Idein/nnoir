import chainer.functions as F

class ConvertBilinear2D():

    def __init__(self, edge, inputs, outputs):
        self.f = lambda x: F.resize_images(x, tuple(edge.params['size']))

    def __call__(self, x):
        return self.f(x)
