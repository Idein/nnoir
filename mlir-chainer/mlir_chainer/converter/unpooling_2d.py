import chainer.functions as F


class ConvertUnpooling2D():

    def __init__(self, function, inputs, outputs):
        self.f = lambda x: F.unpooling_2d(x,
                                          ksize=(function.params['kh'], function.params['kw']),
                                          stride=(function.params['sy'], function.params['sx']),
                                          pad=(function.params['ph'], function.params['pw']),
                                          outsize=(function.params['outh'], function.params['outw']),
                                          cover_all=function.params['cover_all'])

    def __call__(self, x):
        return self.f(x)
