import chainer.functions as F

class ConvertSigmoid():
    def to_chainer(self, edge, x):
        return F.sigmoid(x)
