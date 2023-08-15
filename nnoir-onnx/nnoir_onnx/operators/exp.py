from typing import Any, Dict, List, Optional, Tuple

import onnx
from nnoir.functions import Exp, Function
from numpy.typing import NDArray

from .utils import *


class OpExp(Op):
    def __init__(self, node: onnx.NodeProto, *args: Any):
        super(OpExp, self).__init__(node, *args)

    def to_function(self, env: Dict[str, NDArray[Any]], constants: Dict[str, NDArray[Any]]) -> List[Function]:
        return [Exp(list(self.node.input), list(self.node.output))]
