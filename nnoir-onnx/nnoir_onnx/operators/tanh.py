from typing import Any, Dict, List, Optional, Tuple

import onnx
from nnoir.functions import Function, Tanh
from numpy.typing import NDArray

from .utils import *


class OpTanh(Op):
    def __init__(self, node: onnx.NodeProto, *args: Any):
        super(OpTanh, self).__init__(node, *args)

    def to_function(self, env: Dict[str, NDArray[Any]], constants: Dict[str, NDArray[Any]]) -> List[Function]:
        return [Tanh(list(self.node.input), list(self.node.output))]
