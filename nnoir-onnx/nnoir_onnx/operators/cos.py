from typing import Any, Dict, List, Optional, Tuple

import onnx
from nnoir.functions import Cos, Function
from numpy.typing import NDArray

from .utils import *


class OpCos(Op):
    def __init__(self, node: onnx.NodeProto, *args: Any):
        super(OpCos, self).__init__(node, *args)

    def to_function(self, env: Dict[str, NDArray[Any]], constants: Dict[str, NDArray[Any]]) -> List[Function]:
        return [Cos(list(self.node.input), list(self.node.output))]
