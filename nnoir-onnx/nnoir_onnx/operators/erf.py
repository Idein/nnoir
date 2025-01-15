from typing import Any, Dict, List, Optional, Tuple

import onnx
from nnoir.functions import Erf, Function
from numpy.typing import NDArray

from .utils import *


class OpErf(Op):
    def __init__(self, node: onnx.NodeProto, *args: Any):
        super(OpErf, self).__init__(node, *args)

    def to_function(self, env: Dict[str, NDArray[Any]], constants: Dict[str, NDArray[Any]]) -> List[Function]:
        return [Erf(list(self.node.input), list(self.node.output))]
