import numpy as np
from nnoir.functions import *
from .utils import *


class OpConv(Op):

    def __init__(self, node, *args):
        super(OpConv, self).__init__(node, *args)

        self.kernel_shape = None
        self.auto_pad = b'NOTSET'
        self.pads = None
        self.strides = (1, 1)
        self.dilations = (1, 1)
        self.group = 1
        for attr in node.attribute:
            if attr.name == 'kernel_shape':
                self.kernel_shape = attr.ints
            if attr.name == 'dilations':
                self.dilations = attr.ints
            if attr.name == 'group':
                self.group = attr.i
            if attr.name == 'strides':
                self.strides = attr.ints
            if attr.name == 'auto_pad':
                self.auto_pad = attr.s
            if attr.name == 'pads':
                self.pads = attr.ints

    def to_function(self, env, constants):
        b = None
        if len(self.node.input) == 2:
            [x, W] = self.node.input
        elif len(self.node.input) == 3:
            [x, W, b] = self.node.input
        else:
            raise 'invalid'
        if W not in constants:
            raise UnsupportedONNXOperation(self.node, 'W must be constant')
        W = constants[W]
        if b is not None:
            if b not in constants:
                raise UnsupportedONNXOperation(self.node, 'b must be constant')
            b = constants[b]

        _input = env[x]
        batch = _input.shape[0]
        in_ch = _input.shape[1]
        in_h = _input.shape[2]
        in_w = _input.shape[3]
        kh = self.kernel_shape[0]
        kw = self.kernel_shape[1]
        sy = self.strides[0]
        sx = self.strides[1]
        dy = self.dilations[0]
        dx = self.dilations[1]

        if self.auto_pad == b'NOTSET':
            pad_h = (0, 0)
            pad_w = (0, 0)
            if self.pads is not None:
                pad_h = (self.pads[0], self.pads[2])
                pad_w = (self.pads[1], self.pads[3])
        else:
            pad_h = auto_pad_to_manual_pad(in_h, kh, sy, dy, self.auto_pad)
            pad_w = auto_pad_to_manual_pad(in_w, kw, sx, dx, self.auto_pad)

        is_depthwise = False
        out_channels, in_channels_per_groups = W.shape[:2]
        if self.group > 1 and 1 == in_channels_per_groups:
            return [
                DepthwiseConvolution2D(
                    [x],
                    list(self.node.output),
                    W=np.rollaxis(W.reshape(self.group, out_channels//self.group, kh, kw), 1, 0),
                    b=b,
                    stride=(sy, sx),
                    pad_h=pad_h,
                    pad_w=pad_w,
                    dilate=(dy, dx),
                )
            ]
        else:
            return [
                Convolution2D(
                    [x],
                    list(self.node.output),
                    W=W,
                    b=b,
                    stride=(sy, sx),
                    pad_h=pad_h,
                    pad_w=pad_w,
                    dilate=(dy, dx),
                    groups=self.group
                )
            ]
