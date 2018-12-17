import chainer.functions as F

class ConvertTranspose():

    def __init__(self, edge, inputs, outputs):
        self.f = lambda x: F.transpose(x, edge.params['axes'])

    def __call__(self, x):
        return self.f(x)
