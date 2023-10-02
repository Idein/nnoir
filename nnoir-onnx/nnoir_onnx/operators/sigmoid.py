from typing import Any, Dict, List, Optional, Tuple

import onnx
from nnoir.functions import Function, Sigmoid
from numpy.typing import NDArray

from .utils import *


class OpSigmoid(Op):
    def __init__(self, node: onnx.NodeProto, *args: Any):
        super(OpSigmoid, self).__init__(node, *args)

    def to_function(self, env: Dict[str, NDArray[Any]], constants: Dict[str, NDArray[Any]]) -> List[Function]:
        return [Sigmoid(list(self.node.input), list(self.node.output))]
