import chainer.functions as F

class ConvertClippedReLU():

    def __init__(self, edge, inputs, outputs):
        self.f = lambda x: F.clipped_relu(x, edge.params['upper'])

    def __call__(self, x):
        return self.f(x)
