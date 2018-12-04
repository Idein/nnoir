import chainer.functions as F

class ConvertLinearFunction():
    def to_chainer(edge, x, W, b=None):
        return F.linear(x, W, b)
