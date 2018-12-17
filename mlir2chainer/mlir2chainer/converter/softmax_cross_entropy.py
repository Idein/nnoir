import chainer.functions as F

class ConvertSoftmaxCrossEntropy():

    def __init__(self, edge, inputs, outputs):
        self.f = lambda x: F.softmax_cross_entropy(x, t,
                                                   normalize = edge.params['normalize'],
                                                   cache_score = edge.params['cache_score'])

    def __call__(self, x):
        return self.f(x)
