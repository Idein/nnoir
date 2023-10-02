import functools
from typing import Any, Dict, List, Optional, Tuple

import onnx
from nnoir.functions import Add, Function
from numpy.typing import NDArray

from .utils import *


class OpSum(Op):
    def __init__(self, node: onnx.NodeProto, *args: Any):
        super(OpSum, self).__init__(node, *args)

    def to_function(self, env: Dict[str, NDArray[Any]], constants: Dict[str, NDArray[Any]]) -> List[Function]:
        if len(self.node.input) != 2:
            raise UnsupportedONNXOperation(self.node, "# of inputs must be 2")
        return [Add(list(self.node.input), list(self.node.output))]
