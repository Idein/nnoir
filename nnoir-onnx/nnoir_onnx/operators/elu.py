from typing import Any, Dict, List, Optional, Tuple

import onnx
from nnoir.functions import ELU, Function
from numpy.typing import NDArray

from .utils import *


class OpElu(Op):
    def __init__(self, node: onnx.NodeProto, *args: Any):
        super(OpElu, self).__init__(node, *args)

        self.alpha = 1.0
        for attr in self.node.attribute:
            if attr.name == "alpha":
                self.alpha = attr.f

    def to_function(self, env: Dict[str, NDArray[Any]], constants: Dict[str, NDArray[Any]]) -> List[Function]:
        return [ELU(list(self.node.input), list(self.node.output), alpha=self.alpha)]
