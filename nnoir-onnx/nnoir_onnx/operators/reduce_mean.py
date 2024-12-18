from typing import Any, Dict, List, Optional, Tuple

import onnx
from nnoir.functions import Function, MulConstant, Sum
from numpy.typing import NDArray

from .utils import *


class OpReduceMean(Op):
    def __init__(self, node: onnx.NodeProto, *args: Any):
        super(OpReduceMean, self).__init__(node, *args)

        self.axes = None
        self.keepdims = True
        for attr in node.attribute:
            if attr.name == "axes":
                self.axes = attr.ints
            if attr.name == "keepdims":
                self.keepdims = attr.i > 0

    def to_function(self, env: Dict[str, NDArray[Any]], constants: Dict[str, NDArray[Any]]) -> List[Function]:
        [x] = self.node.input
        [y] = self.node.output
        axes = self.axes
        if axes is None:
            axes = tuple(range(len(env[x].shape)))

        n = 1
        for i in axes:
            n *= env[x].shape[i]

        internal_node = gen_unregisterd_node_name(env)
        register_node(env, internal_node, env[y])

        return [
            Sum(
                list(self.node.input),
                [internal_node],  # type: ignore
                axes=list(axes),
                keepdims=self.keepdims,
            ),
            MulConstant([internal_node], list(self.node.output), value=float(1.0 / n)),  # type: ignore
        ]
