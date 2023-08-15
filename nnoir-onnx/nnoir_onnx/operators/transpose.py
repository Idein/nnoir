from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
from nnoir.functions import Function, Transpose
from numpy.typing import NDArray

from .utils import *


class OpTranspose(Op):
    def __init__(self, node: onnx.NodeProto, *args: Any):
        super(OpTranspose, self).__init__(node, *args)

        self.perm = None
        for attr in node.attribute:
            if attr.name == "perm":
                self.perm = list(attr.ints)

    def to_function(self, env: Dict[str, NDArray[Any]], constants: Dict[str, NDArray[Any]]) -> List[Function]:
        return [Transpose(list(self.node.input), list(self.node.output), axes=self.perm)]
