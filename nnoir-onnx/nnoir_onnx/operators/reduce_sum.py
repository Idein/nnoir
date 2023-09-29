from typing import Any, Dict, List, Optional, Tuple

import onnx
from nnoir.functions import Function, Sum
from numpy.typing import NDArray

from .utils import *


class OpReduceSum(Op):
    def __init__(self, node: onnx.NodeProto, *args: Any):
        super(OpReduceSum, self).__init__(node, *args)

        self.axes = None
        self.keepdims = True
        for attr in node.attribute:
            if attr.name == "axes":
                self.axes = attr.ints
            if attr.name == "keepdims":
                self.keepdims = attr.i > 0

    def to_function(self, env: Dict[str, NDArray[Any]], constants: Dict[str, NDArray[Any]]) -> List[Function]:
        x = self.node.input[0]
        axes = self.axes
        if axes is None and len(self.node.input) > 1:  # Opset 13
            axes = list(env[self.node.input[1]])
        if axes is None:
            axes = tuple(range(len(env[x].shape)))
        return [
            Sum(
                [self.node.input[0]],
                list(self.node.output),
                axes=list(map(int, axes)),
                keepdims=self.keepdims,
            )
        ]
