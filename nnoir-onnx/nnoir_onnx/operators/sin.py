from typing import Any, Dict, List, Optional, Tuple

import onnx
from nnoir.functions import Function, Sin
from numpy.typing import NDArray

from .utils import *


class OpSin(Op):
    def __init__(self, node: onnx.NodeProto, *args: Any):
        super(OpSin, self).__init__(node, *args)

    def to_function(self, env: Dict[str, NDArray[Any]], constants: Dict[str, NDArray[Any]]) -> List[Function]:
        return [Sin(list(self.node.input), list(self.node.output))]
