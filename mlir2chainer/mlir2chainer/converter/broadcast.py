import chainer.functions as F

class ConvertBroadcastTo():
    def to_chainer(self, edge, x):
        return F.broadcast(x)
