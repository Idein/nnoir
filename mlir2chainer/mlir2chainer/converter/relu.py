import chainer.functions as F

class ConvertReLU():
    def to_chainer(edge, x):
        return F.relu(x)
