import chainer.functions as F

class ConvertAdd():

    def __init__(self, edge, inputs, outputs):
        pass

    def __call__(self, *xs):
        return F.add(*xs)
