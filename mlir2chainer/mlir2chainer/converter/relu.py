import chainer.functions as F

class ConvertReLU():
    def to_chainer(self, edge, x):
        return F.relu(x)
