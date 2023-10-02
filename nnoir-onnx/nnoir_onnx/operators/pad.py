from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
from nnoir.functions import ConstantPadding, Function
from numpy.typing import NDArray

from .utils import *


class OpPad(Op):
    def __init__(self, node: onnx.NodeProto, *args: Any):
        super(OpPad, self).__init__(node, *args)

        self.mode = b"constant"
        self.pads = None
        self.value = 0.0

    def to_function(self, env: Dict[str, NDArray[Any]], constants: Dict[str, NDArray[Any]]) -> List[Function]:
        if self.opset_version >= 11:
            # pads
            if not self.node.input[1] in constants:
                raise UnsupportedONNXOperation(self.node, 'pads must be "constant"')
            self.pads = constants[self.node.input[1]]  # type: ignore

            # optional: constant_value
            if len(self.node.input) >= 3:
                if not self.node.input[2] in constants:
                    raise UnsupportedONNXOperation(self.node, 'constant_value must be "constant"')

                v = constants[self.node.input[2]]  # constant_value
                if type(v) == np.ndarray:
                    try:
                        self.value = float(v.item())
                    except ValueError:
                        raise UnsupportedONNXOperation(self.node, "constant_value must be scalar")
                else:
                    self.value = float(v)

            for attr in self.node.attribute:
                if attr.name == "mode":
                    self.mode = attr.s
                else:
                    raise UnsupportedONNXOperation(self.node, f"unknown attribute {attr.s}")

            input_ = [self.node.input[0]]
            pads = list(map(int, self.pads))  # type: ignore # In ONNX specification, the type of `pads` is int64
        else:
            # opset version >= 2
            for attr in self.node.attribute:
                if attr.name == "mode":
                    self.mode = attr.s
                if attr.name == "pads":
                    self.pads = attr.ints
                if attr.name == "value":
                    self.value = attr.f

            input_ = list(self.node.input)
            pads = self.pads  # type: ignore
        if self.mode != b"constant":
            raise UnsupportedONNXOperation(self.node, 'mode must be "constant"')

        n = len(self.pads) // 2  # type: ignore
        return [
            ConstantPadding(
                input_,
                list(self.node.output),
                pads=list(zip(pads[:n], pads[n:])),
                value=self.value,
            )
        ]
