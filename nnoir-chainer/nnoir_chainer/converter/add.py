import chainer.functions as F


class ConvertAdd():

    def __init__(self, function, inputs, outputs):
        pass

    def __call__(self, *xs):
        return F.add(*xs)
