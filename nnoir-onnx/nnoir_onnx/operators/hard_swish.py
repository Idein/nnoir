from typing import Any, Dict, List, Optional, Tuple

import onnx
from nnoir.functions import AddConstant, ClippedReLU, Function, Mul, MulConstant
from numpy.typing import NDArray

from .utils import *


class OpHardSwish(Op):
    def __init__(self, node: onnx.NodeProto, *args: Any):
        super(OpHardSwish, self).__init__(node, *args)

    def to_function(self, env: Dict[str, NDArray[Any]], constants: Dict[str, NDArray[Any]]) -> List[Function]:
        [x] = self.node.input
        t0 = gen_unregisterd_node_name(env)
        register_node(env, t0, env[x])
        t1 = gen_unregisterd_node_name(env)
        register_node(env, t1, env[x])
        t2 = gen_unregisterd_node_name(env)
        register_node(env, t2, env[x])

        return [
            AddConstant([x], [t0], value=3.0),  # type: ignore
            ClippedReLU([t0], [t1], upper=6.0),  # type: ignore
            Mul([x, t1], [t2]),  # type: ignore
            MulConstant([t2], list(self.node.output), value=1 / 6),  # type: ignore
        ]
