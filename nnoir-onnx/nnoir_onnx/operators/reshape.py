from typing import Any, Dict, List, Optional, Tuple

import onnx
from nnoir.functions import Function, Reshape
from numpy.typing import NDArray

from .utils import *


class OpReshape(Op):
    def __init__(self, node: onnx.NodeProto, *args: Any):
        super(OpReshape, self).__init__(node, *args)

    def to_function(self, env: Dict[str, NDArray[Any]], constants: Dict[str, NDArray[Any]]) -> List[Function]:
        [x, _] = self.node.input
        [y] = self.node.output
        return [Reshape([x], list(self.node.output), shape=list(map(int, env[y].shape)))]
