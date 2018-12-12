import chainer.functions as F

class ConvertAdd():
    def to_chainer(self, edge, *xs):
        return F.add(*xs)
