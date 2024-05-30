from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
from nnoir.functions import Function, MaxPooling2D
from numpy.typing import NDArray

from .utils import *


class OpMaxPool(Op):
    def __init__(self, node: onnx.NodeProto, *args: Any):
        super(OpMaxPool, self).__init__(node, *args)

        self.kernel_shape = None
        self.auto_pad = b"NOTSET"
        self.pads = None
        self.storage_order = 0
        self.strides = (1, 1)
        for attr in node.attribute:
            if attr.name == "kernel_shape":
                self.kernel_shape = attr.ints
            elif attr.name == "storage_order":
                self.storage_order = attr.i
            elif attr.name == "strides":
                self.strides = attr.ints
            elif attr.name == "auto_pad":
                self.auto_pad = attr.s
            elif attr.name == "pads":
                self.pads = attr.ints

            # opset version >= 10
            elif attr.name == "ceil_mode":
                if attr.i != 0:
                    raise UnsupportedONNXOperation(self.node, "only value 0 is supported for attribute `ceil_mode`")

            # opset version >= 11
            elif attr.name == "dilations":
                if any(i != 1 for i in attr.ints):
                    raise UnsupportedONNXOperation(self.node, "only array of 1 is supported for attribute `dilations`")
            else:
                raise UnsupportedONNXOperation(self.node, f"unknown attribute {attr.name}")

    def to_function(self, env: Dict[str, NDArray[Any]], constants: Dict[str, NDArray[Any]]) -> List[Function]:
        [x] = self.node.input

        _input = env[x]
        batch = _input.shape[0]
        channel = _input.shape[1]
        in_h = _input.shape[2]
        in_w = _input.shape[3]
        kh = self.kernel_shape[0]  # type: ignore
        kw = self.kernel_shape[1]  # type: ignore
        sy = self.strides[0]
        sx = self.strides[1]

        if self.auto_pad == b"NOTSET":
            pad_h = (0, 0)
            pad_w = (0, 0)
            if self.pads is not None:
                pad_h = (self.pads[0], self.pads[2])
                pad_w = (self.pads[1], self.pads[3])
        else:
            pad_h = auto_pad_to_manual_pad(in_h, kh, sy, 1, self.auto_pad)
            pad_w = auto_pad_to_manual_pad(in_w, kw, sx, 1, self.auto_pad)

        return [
            MaxPooling2D(
                list(self.node.input),
                list(self.node.output),
                kernel=(kh, kw),
                stride=(sy, sx),
                pad_h=pad_h,
                pad_w=pad_w,
            )
        ]
