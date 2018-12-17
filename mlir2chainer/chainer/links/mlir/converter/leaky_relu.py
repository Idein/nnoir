import chainer.functions as F

class ConvertLeakyReLU():

    def __init__(self, edge, inputs, outputs):
        self.f = lambda x: F.leaky_relu(x, edge.params['slope'])

    def __call__(self, x):
        return self.f(x)
