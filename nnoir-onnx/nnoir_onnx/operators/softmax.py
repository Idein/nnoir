from typing import Any, Dict, List, Optional, Tuple

import onnx
from nnoir.functions import Function, Softmax
from numpy.typing import NDArray

from .utils import *


class OpSoftmax(Op):
    def __init__(self, node: onnx.NodeProto, *args: Any):
        super(OpSoftmax, self).__init__(node, *args)

        self.axis = -1 if self.opset_version >= 13 else 1
        for attr in self.node.attribute:
            if attr.name == "axis":
                self.axis = attr.i

    def to_function(self, env: Dict[str, NDArray[Any]], constants: Dict[str, NDArray[Any]]) -> List[Function]:
        [x] = self.node.input
        if self.axis < 0:
            self.axis = env[x].ndim + self.axis

        return [Softmax(list(self.node.input), list(self.node.output), axis=self.axis)]
