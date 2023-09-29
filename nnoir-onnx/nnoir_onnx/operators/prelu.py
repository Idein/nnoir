from typing import Any, Dict, List, Optional, Tuple

import onnx
from nnoir.functions import Function, LeakyReLU
from numpy.typing import NDArray

from .utils import *


class OpPRelu(Op):
    def __init__(self, node: onnx.NodeProto, *args: Any):
        super(OpPRelu, self).__init__(node, *args)

    def to_function(self, env: Dict[str, NDArray[Any]], constants: Dict[str, NDArray[Any]]) -> List[Function]:
        [x, slope] = self.node.input

        if slope in constants.keys():
            c = constants[slope].ravel()
            if len(c) != 1:
                raise UnsupportedONNXOperation(self.node, "# of slope size must be 1")
            v: List[Function] = [LeakyReLU([x], list(self.node.output), slope=float(c[0]))]
            return v
        else:
            raise UnsupportedONNXOperation(self.node, "# of slope must be constant")
