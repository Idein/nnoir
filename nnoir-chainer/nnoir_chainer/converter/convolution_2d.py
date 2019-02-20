import numpy
import chainer
import chainer.links as L
import chainer.functions as F
import chainer.utils


class ConvertConvolution2D(chainer.Chain):

    def __init__(self, function, inputs, outputs):
        super(ConvertConvolution2D, self).__init__()
        [kh, kw] = function.params['W'].shape[2:]
        [sy, sx] = function.params['stride']
        [in_h, in_w] = inputs[0].shape[2:]
        [out_h, out_w] = outputs[0].shape[2:]
        [ph_pre, ph_post] = function.params['pad_h']
        [pw_pre, pw_post] = function.params['pad_w']
        ph_post = max(ph_post - ((ph_pre + in_h + ph_post) - ((out_h - 1) * sy + kh)), 0)
        pw_post = max(pw_post - ((pw_pre + in_w + pw_post) - ((out_w - 1) * sx + kw)), 0)
        padding = [(0, 0), (0, 0), (ph_pre, ph_post), (pw_pre, pw_post)]
        self.pad = lambda x: F.pad(x, padding, mode='constant', constant_values=0.0)
        with self.init_scope():
            self.conv = L.Convolution2D(in_channels=function.params['W'].shape[1] * function.params['groups'],
                                        out_channels=function.params['W'].shape[0],
                                        ksize=tuple(function.params['W'].shape[2:]),
                                        stride=tuple(function.params['stride']),
                                        pad=0,
                                        nobias=(function.params['b'] is None),
                                        initialW=function.params['W'],
                                        initial_bias=function.params['b'],
                                        dilate=tuple(function.params['dilate']),
                                        groups=function.params['groups'])

    def __call__(self, x):
        return self.conv(self.pad(x))
