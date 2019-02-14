import chainer.functions as F


class ConvertSoftmaxCrossEntropy():

    def __init__(self, function, inputs, outputs):
        self.f = lambda x: F.softmax_cross_entropy(x, t,
                                                   normalize=function.params['normalize'],
                                                   cache_score=function.params['cache_score'])

    def __call__(self, x):
        return self.f(x)
