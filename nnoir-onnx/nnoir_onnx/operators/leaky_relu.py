from typing import Any, Dict, List, Optional, Tuple

import onnx
from nnoir.functions import Function, LeakyReLU
from numpy.typing import NDArray

from .utils import *


class OpLeakyRelu(Op):
    def __init__(self, node: onnx.NodeProto, *args: Any):
        super(OpLeakyRelu, self).__init__(node, *args)

        self.alpha = 0.01
        for attr in node.attribute:
            if attr.name == "alpha":
                self.alpha = attr.f

    def to_function(self, env: Dict[str, NDArray[Any]], constants: Dict[str, NDArray[Any]]) -> List[Function]:
        return [LeakyReLU(list(self.node.input), list(self.node.output), slope=self.alpha)]
