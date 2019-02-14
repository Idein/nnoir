import chainer.functions as F


class ConvertDropout():

    def __init__(self, function, inputs, outputs):
        pass

    def __call__(self, x):
        return F.dropout(x)
