from typing import Any, Dict, List, Optional, Tuple

import onnx
from nnoir.functions import Function, ReLU
from numpy.typing import NDArray

from .utils import *


class OpRelu(Op):
    def __init__(self, node: onnx.NodeProto, *args: Any):
        super(OpRelu, self).__init__(node, *args)

    def to_function(self, env: Dict[str, NDArray[Any]], constants: Dict[str, NDArray[Any]]) -> List[Function]:
        return [ReLU(list(self.node.input), list(self.node.output))]
