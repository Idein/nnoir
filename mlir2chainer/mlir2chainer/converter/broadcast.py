import chainer.functions as F

class ConvertBroadcastTo():
    def to_chainer(edge, x):
        return F.broadcast(x)
