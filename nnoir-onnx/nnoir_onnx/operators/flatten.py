from functools import reduce
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
from nnoir.functions import Function, Reshape
from numpy.typing import NDArray

from .utils import *


class OpFlatten(Op):
    def __init__(self, node: onnx.NodeProto, *args: Any):
        super(OpFlatten, self).__init__(node, *args)

        self.axis = 1
        for attr in node.attribute:
            if attr.name == "axis":
                self.axis = attr.i

    def to_function(self, env: Dict[str, NDArray[Any]], constants: Dict[str, NDArray[Any]]) -> List[Function]:
        [x] = self.node.input
        if self.axis == 0:
            flattened_shape = (1, -1)
        else:
            flattened_shape = (
                reduce(lambda k, n: k * n, env[x].shape[: self.axis]),
                reduce(lambda k, n: k * n, env[x].shape[self.axis :]),
            )
        return [Reshape(list(self.node.input), list(self.node.output), shape=flattened_shape)]
