import chainer.functions as F

class ConvertConcat():

    def __init__(self, edge, inputs, outputs):
        self.f = lambda xs: F.concat(xs, edge.params['axis'])

    def __call__(self, *xs):
        return self.f(xs)
