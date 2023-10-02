from typing import Any, Dict, List, Optional, Tuple

import onnx
from nnoir.functions import Function, Tan
from numpy.typing import NDArray

from .utils import *


class OpTan(Op):
    def __init__(self, node: onnx.NodeProto, *args: Any):
        super(OpTan, self).__init__(node, *args)

    def to_function(self, env: Dict[str, NDArray[Any]], constants: Dict[str, NDArray[Any]]) -> List[Function]:
        return [Tan(list(self.node.input), list(self.node.output))]
