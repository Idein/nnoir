import numpy
import six
import chainer.links as L

class ConvertDepthwiseConvolution2D():

    def __init__(self, edge, inputs, outputs):
        # TODO: input/output の shape をから，padding量を計算
        padding = (edge.params['pad_h'][0], edge.params['pad_w'][0]),
        self.pad = lambda x: x
        if edge.params['pad_h'][0] != edge.params['pad_h'][1] or edge.params['pad_w'][0] != edge.params['pad_w'][1]:
            padding = 0
            self.pad = lambda x: F.pad(x, pad, mode='constants', constant_values=0)
        in_channels = inputs[0].shape[1]
        groups = edge.params['W'].shape[1]
        out_channels = edge.params['W'].shape[0] * groups
        kh = edge.params['W'].shape[2]
        kw = edge.params['W'].shape[3]
        self.conv = L.Convolution2D(in_channels = groups,
                                    out_channels = out_channels,
                                    ksize = (kh, kw),
                                    stride = tuple(edge.params['stride']),
                                    pad = (edge.params['pad_h'][0], edge.params['pad_w'][0]),
                                    nobias = (edge.params['b'] is None),
                                    initialW = numpy.rollaxis(edge.params['W'], 1, 0).reshape(out_channels, in_channels//groups, kh, kw),
                                    initial_bias = edge.params['b'],
                                    dilate = edge.params['dilate'],
                                    groups = groups)

    def __call__(self, x):
        return self.conv(self.pad(x))
