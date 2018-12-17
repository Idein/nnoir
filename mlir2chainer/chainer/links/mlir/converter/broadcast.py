import chainer.functions as F

class ConvertBroadcastTo():

    def __init__(self, edge, inputs, outputs):
        pass

    def __call__(self, x):
        return F.broadcast(x)
