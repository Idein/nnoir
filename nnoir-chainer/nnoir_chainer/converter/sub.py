import chainer.functions.math.basic_math as F


class ConvertSub():

    def __init__(self, function, inputs, outputs):
        pass

    def __call__(self, *xs):
        return F.sub(*xs)
