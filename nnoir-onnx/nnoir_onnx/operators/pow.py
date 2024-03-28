from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
from nnoir.functions import Function, Mul
from numpy.typing import NDArray

from .utils import *


class OpPow(Op):
    def __init__(self, node: onnx.NodeProto, *args: Any):
        super(OpPow, self).__init__(node, *args)

    def to_function(self, env: Dict[str, NDArray[Any]], constants: Dict[str, NDArray[Any]]) -> List[Function]:
        [a, b] = self.node.input

        if b in constants and constants[b] == 2.0:
            return [Mul([a, a], list(self.node.output))]
        else:
            raise UnsupportedONNXOperation(self.node, "unimplemented yet")
