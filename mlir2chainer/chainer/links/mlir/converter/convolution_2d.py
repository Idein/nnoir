import numpy
import six
import chainer.links as L
import chainer.functions as F
import chainer.utils

class ConvertConvolution2D():

    def __init__(self, edge, inputs, outputs):
        [kh, kw] = edge.params['W'].shape[2:]
        [sy, sx] = edge.params['stride']
        [in_h, in_w] = inputs[0].shape[2:]
        [out_h, out_w] = outputs[0].shape[2:]
        [ph_pre, ph_post] = edge.params['pad_h']
        [pw_pre, pw_post] = edge.params['pad_w']
        ph_post = max(ph_post - ((ph_pre + in_h + ph_post) - ((out_h - 1) * sy + kh)), 0)
        pw_post = max(pw_post - ((pw_pre + in_w + pw_post) - ((out_w - 1) * sx + kw)), 0)
        padding = [(0,0),(0,0),(ph_pre,ph_post),(pw_pre,pw_post)]
        self.pad = lambda x: F.pad(x, padding, mode='constant', constant_values=0.0)
        self.conv = L.Convolution2D(in_channels = edge.params['W'].shape[1] * edge.params['groups'],
                                    out_channels = edge.params['W'].shape[0],
                                    ksize = tuple(edge.params['W'].shape[2:]),
                                    stride = tuple(edge.params['stride']),
                                    pad = 0,
                                    nobias = (edge.params['b'] is None),
                                    initialW = edge.params['W'],
                                    initial_bias = edge.params['b'],
                                    dilate = tuple(edge.params['dilate']),
                                    groups = edge.params['groups'])

    def __call__(self, x):
        return self.conv(self.pad(x))
