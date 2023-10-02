from typing import Any, Dict, List, Optional, Tuple

import onnx
from nnoir.functions import AddConstant, ClippedReLU, Function, MulConstant
from numpy.typing import NDArray

from .utils import *


# https://github.com/onnx/onnx/blob/main/docs/Operators.md#hardsigmoid
class OpHardSigmoid(Op):
    def __init__(self, node: onnx.NodeProto, *args: Any):
        super(OpHardSigmoid, self).__init__(node, *args)
        if self.opset_version < 6:
            raise UnsupportedONNXOperation(self.node, "only opset_version >= 6 is supported")

    def to_function(self, env: Dict[str, NDArray[Any]], constants: Dict[str, NDArray[Any]]) -> List[Function]:
        [x] = self.node.input
        t0 = gen_unregisterd_node_name(env)
        register_node(env, t0, env[x])
        t1 = gen_unregisterd_node_name(env)
        register_node(env, t1, env[x])

        alpha = 0.2
        beta = 0.5
        for attr in self.node.attribute:
            if attr.name == "alpha":
                alpha = attr.f
            if attr.name == "beta":
                beta = attr.f

        return [
            # t0 = alpha * x
            MulConstant([x], [t0], value=alpha),  # type: ignore
            # t1 = alpha * x + beta
            AddConstant([t0], [t1], value=beta),  # type: ignore
            # output = max(0, min(alpha * x + beta, 1))
            ClippedReLU([t1], list(self.node.output), upper=1.0),  # type: ignore
        ]
