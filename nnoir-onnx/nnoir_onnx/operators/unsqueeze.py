from typing import Any, Dict, List, Optional, Tuple

import onnx
from nnoir.functions import Function, Reshape
from numpy.typing import NDArray

from .utils import *


class OpUnsqueeze(Op):
    def __init__(self, node: onnx.NodeProto, *args: Any):
        super(OpUnsqueeze, self).__init__(node, *args)

    def to_function(self, env: Dict[str, NDArray[Any]], constants: Dict[str, NDArray[Any]]) -> List[Function]:
        x = self.node.input[0]
        [y] = self.node.output

        # The axes attribute is ignored. We already know output shape,
        # without reconstruction from input shape and axes.

        return [Reshape([x], list(self.node.output), shape=list(map(int, env[y].shape)))]
