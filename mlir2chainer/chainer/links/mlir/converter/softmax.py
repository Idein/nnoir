import chainer.functions as F

class ConvertSoftmax():

    def __init__(self, edge, inputs, outputs):
        self.f = lambda x: F.softmax(x, edge.params['axis'])

    def __call__(self, x):
        return self.f(x)
