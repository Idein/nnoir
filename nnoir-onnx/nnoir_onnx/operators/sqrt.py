from typing import Any, Dict, List, Optional, Tuple

import onnx
from nnoir.functions import Function, Sqrt
from numpy.typing import NDArray

from .utils import *


class OpSqrt(Op):
    def __init__(self, node: onnx.NodeProto, *args: Any):
        super(OpSqrt, self).__init__(node, *args)

    def to_function(self, env: Dict[str, NDArray[Any]], constants: Dict[str, NDArray[Any]]) -> List[Function]:
        return [Sqrt(list(self.node.input), list(self.node.output))]
