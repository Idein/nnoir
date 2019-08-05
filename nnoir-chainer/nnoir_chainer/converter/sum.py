import numpy
import chainer
import chainer.links as L
import chainer.functions as F
import chainer.utils


class ConvertSum(chainer.Chain):

    def __init__(self, function, inputs, outputs):
        super(ConvertSum, self).__init__()
        axes = function.params['axes']
        shape = inputs[0].shape
        if axes is None:
            axes = tuple(range(len(shape)))
        if type(axes) is int:
            axes = (axes, )
        axes = tuple(axes)
        keepdims = function.params['keepdims']
        self.f = lambda x: F.sum(x, axis=axes, keepdims=keepdims)

    def __call__(self, x):
        return self.f(x)
