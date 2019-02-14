import chainer.functions as F


class ConvertLocalResponseNormalization():

    def __init__(self, function, inputs, outputs):
        self.f = lambda x: F.local_response_normalization(x,
                                                          function.params['n'],
                                                          function.params['k'],
                                                          function.params['alpha'],
                                                          function.params['beta'])

    def __call__(self, x):
        return self.f(x)
