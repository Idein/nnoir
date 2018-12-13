import chainer.functions as F

class ConvertMul():
    def to_chainer(self, edge, lhs, rhs):
        return lhs * rhs
