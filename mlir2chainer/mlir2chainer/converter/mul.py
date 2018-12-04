import chainer.functions as F

class ConvertMul():
    def to_chainer(edge, lhs, rhs):
        return lhs * rhs
