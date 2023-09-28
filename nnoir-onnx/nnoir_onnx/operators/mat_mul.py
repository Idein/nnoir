from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
from nnoir.functions import Constant, Function, Linear, MatMul
from numpy.typing import NDArray

from .utils import Op, gen_unregisterd_node_name, register_node


def gen_value(env: Dict[str, NDArray[Any]], arr: NDArray[Any]) -> str:
    name = gen_unregisterd_node_name(env)
    register_node(env, name, arr)

    return name


class OpMatMul(Op):
    def __init__(self, node: onnx.NodeProto, *args: Any):
        super(OpMatMul, self).__init__(node, *args)

    def to_function(self, env: Dict[str, NDArray[Any]], constants: Dict[str, NDArray[Any]]) -> List[Function]:
        [x, W] = self.node.input
        if W in constants and constants[W].ndim == 2 and env[x].ndim == 2:
            return [Linear([x], list(self.node.output), W=env[W].T, b=None)]
        elif W in constants:
            const_name = gen_value(env, constants[W])
            nodes = [
                Constant([], [const_name], value=constants[W]),  # type: ignore
                MatMul([x, const_name], list(self.node.output)),  # type: ignore
            ]
            return nodes
        else:
            return [MatMul(list(self.node.input), list(self.node.output))]
