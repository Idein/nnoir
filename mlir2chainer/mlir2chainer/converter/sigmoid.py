import chainer.functions as F

class ConvertSigmoid():
    def to_chainer(edge, x):
        return F.sigmoid(x)
