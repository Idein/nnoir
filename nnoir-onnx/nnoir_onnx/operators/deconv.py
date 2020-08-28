import numpy as np
import nnoir
from nnoir.functions import Deconvolution2D, Convolution2D
from .utils import Op, UnsupportedONNXOperation, auto_pad_to_manual_pad

from nnoir.functions.util import get_deconv_outsize


class OpConvTranspose(Op):
    def __init__(self, node, *args):
        super(OpConvTranspose, self).__init__(node, *args)

        self.kernel_shape = None
        self.auto_pad = b'NOTSET'
        self.pads = None
        self.strides = (1, 1)
        self.dilations = (1, 1)
        self.group = 1
        self.output_shape = None
        # self.output_padding = None

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
            if attr.name == 'output_padding':
                raise UnsupportedONNXOperation(self.node, 'output_padding is not surporrted')
                # self.output_padding = attr.ints
            if attr.name == 'output_shape':
                self.output_shape = attr.ints

                # The onnx doc say if `output_shape` is specified, pads are ignored
                self.pads = None

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

        # `kernel_shape` may be ommited. Then, it should be inferred.
        if self.kernel_shape is None:
            (_, _, kh, kw) = W.shape
        else:
            kh = self.kernel_shape[0]
            kw = self.kernel_shape[1]

        _input = env[x]
        in_h = _input.shape[2]
        in_w = _input.shape[3]
        sy = self.strides[1]
        sx = self.strides[0]
        dy = self.dilations[1]
        dx = self.dilations[0]

        if self.output_shape is not None:
            if sx == sy == dx == dy == 1:
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

            raise UnsupportedONNXOperation(self.node, 'output_shape is not surporrted')

        else:
            if self.auto_pad == b'NOTSET':
                pad_h = (0, 0)
                pad_w = (0, 0)
                if self.pads is not None:
                    pad_h = (self.pads[0], self.pads[2])
                    pad_w = (self.pads[1], self.pads[3])
            else:
                pad_h, pad_w = calc_output_padding(in_h, in_w, kh, kw, sx, sy, dx, dy)
                print(pad_h)
                print(pad_w)
            return [
                Deconvolution2D(
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


def calc_output_padding(in_h, in_w, kh, kw, sx, sy, dx, dy):
    '''
    For attribute `output_shape`. Currently unused.
    '''
    output_padding = [0, 0]

    out_h = get_deconv_outsize(in_h, kh, sx, dx, (0, 0), (0, 0))
    out_w = get_deconv_outsize(in_w, kw, sy, dy, (0, 0), (0, 0))

    output_padding[0] = self.output_shape[0] - out_h
    output_padding[1] = self.output_shape[1] - out_w

    assert output_padding[0] >= 0 or output_padding[1] >= 0

    total_padding = [0, 0]

    total_padding[0] = sx * (in_h - 1) + output_padding[0] + ((kh - 1) * dx + 1) - out_h
    total_padding[1] = sy * (in_w - 1) + output_padding[1] + ((kw - 1) * dy + 1) - out_w

    if (self.auto_pad != b'SAME_UPPER'):
        pad_h = (total_padding[0] // 2, total_padding[0] - (total_padding[0] // 2))
        pad_w = (total_padding[1] // 2, total_padding[1] - (total_padding[1] // 2))
    else:
        pad_h = (total_padding[0] - (total_padding[0] // 2), total_padding[0] // 2)
        pad_w = (total_padding[1] - (total_padding[1] // 2), total_padding[1] // 2)

    return (pad_h, pad_w)
