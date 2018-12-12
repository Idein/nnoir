import numpy
import six
import chainer.links as L

class ConvertConvolution2D():
    def to_chainer(self, edge, x):
        if edge.params['pad_h'][0] != edge.params['pad_h'][1] or edge.params['pad_w'][0] != edge.params['pad_w'][1]:
            raise Exception('this padding is not supported now')
        conv = L.Convolution2D(in_channels = edge.params['W'].shape[1],
                               out_channels = edge.params['W'].shape[0],
                               ksize = (edge.params['W'].shape[2], edge.params['W'].shape[3]),
                               stride = tuple(edge.params['stride']),
                               pad = (edge.params['pad_h'][0], edge.params['pad_w'][0]),
                               nobias = (edge.params['b'] is None),
                               dilate = tuple(edge.params['dilate']),
                               groups = edge.params['groups'])
        conv.W.data = edge.params['W']
        if edge.params['b'] is not None:
            conv.b.data = edge.params['b']
        return conv(x)
