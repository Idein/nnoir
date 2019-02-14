import numpy
import chainer
import chainer.links as L


class ConvertDepthwiseConvolution2D(chainer.Chain):

    def __init__(self, function, inputs, outputs):
        super(ConvertDepthwiseConvolution2D, self).__init__()
        padding = (function.params['pad_h'][0], function.params['pad_w'][0]),
        self.pad = lambda x: x
        if function.params['pad_h'][0] != function.params['pad_h'][1] or function.params['pad_w'][0] != function.params['pad_w'][1]:
            padding = 0
            self.pad = lambda x: F.pad(x, pad, mode='constants', constant_values=0)
        in_channels = inputs[0].shape[1]
        groups = function.params['W'].shape[1]
        out_channels = function.params['W'].shape[0] * groups
        kh = function.params['W'].shape[2]
        kw = function.params['W'].shape[3]
        with self.init_scope():
            self.conv = L.Convolution2D(in_channels=groups,
                                        out_channels=out_channels,
                                        ksize=(kh, kw),
                                        stride=tuple(function.params['stride']),
                                        pad=(function.params['pad_h'][0], function.params['pad_w'][0]),
                                        nobias=(function.params['b'] is None),
                                        initialW=numpy.rollaxis(function.params['W'], 1, 0).reshape(
                                            out_channels, in_channels//groups, kh, kw),
                                        initial_bias=function.params['b'],
                                        dilate=function.params['dilate'],
                                        groups=groups)

    def __call__(self, x):
        return self.conv(self.pad(x))
