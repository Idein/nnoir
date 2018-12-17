import chainer.functions as F

class ConvertReshape():

    def __init__(self, edge, inputs, outputs):
        self.f = lambda x: F.reshape(x, tuple(edge.params['shape']))

    def __call__(self, x):
        return self.f(x)
