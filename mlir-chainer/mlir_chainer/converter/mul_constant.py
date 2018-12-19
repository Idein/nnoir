import chainer.functions as F

class ConvertMulConstant():

    def __init__(self, edge, inputs, outputs):
        self.f = lambda x: x * edge.params['value']

    def __call__(self, x):
        return self.f(x)
