import chainer.functions as F

class ConvertConvolution2DFunction():
    def to_chainer(edge, x, W, b=None):
        if edge.params['pad_h'][0] != edge.params['pad_h'][1] or edge.params['pad_w'][0] != edge.params['pad_w'][1]:
            raise Exception('this padding is not supported now')
        return F.convolution_2d(x, W, b,
                                stride = tuple(edge.params['stride']),
                                pad = (edge.params['pad_h'][0], edge.params['pad_w'][0]),
                                dilate = tuple(edge.params['dilate']),
                                groups = edge.params['groups'])
