from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
from nnoir.functions import Concat, Function
from numpy.typing import NDArray

from .utils import *


class OpConcat(Op):
    def __init__(self, node: onnx.NodeProto, *args: Any):
        super(OpConcat, self).__init__(node, *args)

        self.axis = None
        for attr in node.attribute:
            if attr.name == "axis":
                self.axis = attr.i

    def to_function(self, env: Dict[str, NDArray[Any]], constants: Dict[str, NDArray[Any]]) -> List[Function]:
        [y] = self.node.output
        axis = len(env[y].shape) + self.axis if self.axis < 0 else self.axis  # type: ignore
        return [Concat(list(self.node.input), list(self.node.output), axis=int(axis))]  # type: ignore
