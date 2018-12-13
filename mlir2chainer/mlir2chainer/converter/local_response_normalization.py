import chainer.functions as F

class ConvertLocalResponseNormalization():
    def to_chainer(self, edge, x):
        return F.local_response_normalization(x,
                                              edge.params['n'],
                                              edge.params['k'],
                                              edge.params['alpha'],
                                              edge.params['beta'])
