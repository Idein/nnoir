import chainer.functions as F

class ConvertSigmoid():

    def __init__(self, edge, inputs, outputs):
        pass

    def __call__(self, x):
        return F.sigmoid(x)
