import chainer.functions as F

class ConvertConvolution2DFunction():
    def to_chainer(edge, x, W, b=None):
        return F.convolution_2d(x, W, b)
