import chainer.functions as F

class ConvertDropout():
    def to_chainer(edge, x):
        return F.dropout(x)
