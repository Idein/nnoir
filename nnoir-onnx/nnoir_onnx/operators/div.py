from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
from nnoir.functions import Constant, Div, Function, Mul
from numpy.typing import NDArray

from .utils import *


class OpDiv(Op):
    def __init__(self, node: onnx.NodeProto, *args: Any):
        super(OpDiv, self).__init__(node, *args)

    def to_function(self, env: Dict[str, NDArray[Any]], constants: Dict[str, NDArray[Any]]) -> List[Function]:
        [a, b] = self.node.input

        def scale(v: str, w: NDArray[Any]) -> List[Function]:
            internal_node = gen_unregisterd_node_name(env)
            register_node(env, internal_node, w)
            return [
                Constant([], [internal_node], value=w),  # type: ignore
                Mul([v, internal_node], list(self.node.output)),  # type: ignore
            ]

        if a in constants and b not in constants:
            raise UnsupportedONNXOperation(self.node, "unimplemented yet")
        elif a not in constants and b in constants:
            c = constants[b]
            if type(c) == np.ndarray and c.shape != ():  # usual ndarray
                w = 1 / c
            elif type(c) == np.ndarray and c.shape == ():  # ndarray with 0-dimension
                w = 1 / np.array([float(c)], dtype=np.float32)
            else:
                raise UnsupportedONNXOperation(self.node, "bug! (unreachable here)")

            assert type(w) == np.ndarray and w.shape != ()
            return scale(a, w)

        elif a not in constants and b not in constants:
            return [Div(list(self.node.input), list(self.node.output))]
        else:
            raise UnsupportedONNXOperation(self.node, "bug! (unreachable here)")
