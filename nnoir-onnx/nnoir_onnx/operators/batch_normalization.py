from typing import Any, Dict, List, Optional, Tuple

import onnx
from nnoir.functions import BatchNormalization, Function
from numpy.typing import NDArray

from .utils import *


class OpBatchNormalization(Op):
    def __init__(self, node: onnx.NodeProto, *args: Any):
        super().__init__(node, *args)

    def to_function(self, env: Dict[str, NDArray[Any]], constants: Dict[str, NDArray[Any]]) -> List[Function]:
        [x, gamma, beta, mean, var] = self.node.input
        if gamma not in constants:
            raise UnsupportedONNXOperation(self.node, "missing gamma")
        if beta not in constants:
            raise UnsupportedONNXOperation(self.node, "missing beta")
        if mean not in constants:
            raise UnsupportedONNXOperation(self.node, "missing mean")
        if var not in constants:
            raise UnsupportedONNXOperation(self.node, "missing var")
        eps = 1e-05
        for attr in self.node.attribute:
            if attr.name == "epsilon":
                eps = attr.f
        return [
            BatchNormalization(
                [x],
                list(self.node.output),
                eps=eps,
                avg_mean=constants[mean],
                avg_var=constants[var],
                gamma=constants[gamma],
                beta=constants[beta],
            )
        ]
