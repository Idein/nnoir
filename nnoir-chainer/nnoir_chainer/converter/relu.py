import chainer.functions as F


class ConvertReLU:
    def __init__(self, function, inputs, outputs):
        pass

    def __call__(self, x):
        return F.relu(x)
