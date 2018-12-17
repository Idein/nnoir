import chainer.functions as F

class ConvertMul():

    def __init__(self, edge, inputs, outputs):
        pass

    def __call__(self, lhs, rhs):
        return lhs * rhs
