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
                raise UnsupportedONNXOperation(self.node, 'output_padding is not supported')
            if attr.name == 'output_shape':
                self.output_shape = attr.ints

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
        sy = self.strides[0]
        sx = self.strides[1]
        dy = self.dilations[0]
        dx = self.dilations[1]

        if self.output_shape is not None:
            # if output_shape is apecified, we must infer `pads` by the following equations as follows:
            # https://github.com/onnx/onnx/blob/master/docs/Operators.md#ConvTranspose
            def infer_total_pad(in_size, s, k, d, out_size):
                return s * (in_size - 1) + ((k - 1) * d + 1) - out_size

            def infer_pad(auto_pads: bytes, total_padding):
                if (auto_pads != b'SAME_UPPER'):
                    return (total_padding // 2, total_padding - (total_padding // 2))
                else:
                    return (total_padding - (total_padding // 2), total_padding // 2)

            _, _, in_h, in_w = _input.shape
            _, _, kh, kw = W.shape

            total_pad_h = infer_total_pad(in_h, sy, kh, dy, self.output_shape[0])
            pad_h = infer_pad(self.auto_pad, total_pad_h)
            total_pad_w = infer_total_pad(in_w, sx, kw, dx, self.output_shape[1])
            pad_w = infer_pad(self.auto_pad, total_pad_w)
        else:
            if self.auto_pad == b'NOTSET':
                pad_h = (0, 0)
                pad_w = (0, 0)
                if self.pads is not None:
                    pad_h = (self.pads[0], self.pads[2])
                    pad_w = (self.pads[1], self.pads[3])
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
