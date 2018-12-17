import chainer.functions as F

class ConvertLocalResponseNormalization():

    def __init__(self, edge, inputs, outputs):
        self.f = lambda x: F.local_response_normalization(x,
                                                          edge.params['n'],
                                                          edge.params['k'],
                                                          edge.params['alpha'],
                                                          edge.params['beta'])

    def __call__(self, x):
        return self.f(x)
