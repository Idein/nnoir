import chainer.functions as F

class ConvertDropout():
    def to_chainer(self, edge, x):
        return F.dropout(x)
