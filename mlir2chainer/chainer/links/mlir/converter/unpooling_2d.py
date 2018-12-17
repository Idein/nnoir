import chainer.functions as F

class ConvertUnpooling2D():

    def __init__(self, edge, inputs, outputs):
        self.f = lambda x: F.unpooling_2d(x,
                                          ksize = (edge.params['kh'], edge.params['kw']),
                                          stride = (edge.params['sy'], edge.params['sx']),
                                          pad = (edge.params['ph'], edge.params['pw']),
                                          outsize = (edge.params['outh'], edge.params['outw']),
                                          cover_all = edge.params['cover_all'])

    def __call__(self, x):
        return self.f(x)
