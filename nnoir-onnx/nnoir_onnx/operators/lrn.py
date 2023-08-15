from typing import Any, Dict, List, Optional, Tuple

import onnx
from nnoir.functions import Function, LocalResponseNormalization
from numpy.typing import NDArray

from .utils import *


class OpLRN(Op):
    def __init__(self, node: onnx.NodeProto, *args: Any):
        super(OpLRN, self).__init__(node, *args)

        self.alpha = 0.0001
        self.beta = 0.75
        self.bias = 1.0
        self.size = None
        for attr in node.attribute:
            if attr.name == "alpha":
                self.alpha = attr.f
            if attr.name == "beta":
                self.beta = attr.f
            if attr.name == "bias":
                self.bias = attr.f
            if attr.name == "size":
                self.size = attr.i

    def to_function(self, env: Dict[str, NDArray[Any]], constants: Dict[str, NDArray[Any]]) -> List[Function]:
        return [
            LocalResponseNormalization(
                list(self.node.input),
                list(self.node.output),
                n=self.size,
                k=self.bias,
                alpha=self.alpha / self.size,  # type: ignore
                beta=self.beta,
            )
        ]
