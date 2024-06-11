from typing import Any, Dict, List

import onnx
from nnoir.functions import Bias, Constant, Function, Sub
from numpy.typing import NDArray

from .utils import *


class OpSub(Op):
    def __init__(self, node: onnx.NodeProto, *args: Any):
        super(OpSub, self).__init__(node, *args)

    def to_function(self, env: Dict[str, NDArray[Any]], constants: Dict[str, NDArray[Any]]) -> List[Function]:
        [a, b] = self.node.input
        if a in constants and b not in constants:
            internal_node = gen_unregisterd_node_name(env)
            register_node(env, internal_node, env[a])
            return [
                Constant([], [internal_node], value=constants[a]),  # type: ignore
                Sub([internal_node, b], list(self.node.output)),  # type: ignore
            ]
        elif a not in constants and b in constants:
            return [Bias([a], list(self.node.output), axis=0, b=encode_ndarray(-constants[b]))]
        elif a not in constants and b not in constants:
            return [Sub(list(self.node.input), list(self.node.output))]
        else:
            raise UnsupportedONNXOperation(self.node, "bug! (unreachable here)")
