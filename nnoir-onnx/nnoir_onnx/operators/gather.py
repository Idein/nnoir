from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
from nnoir.functions import Function, Reshape, Slice
from numpy.typing import NDArray

from .utils import *


class OpGather(Op):
    def __init__(self, node: onnx.NodeProto, *args: Any):
        super(OpGather, self).__init__(node, *args)

        self.axis = 0
        for attr in node.attribute:
            if attr.name == "axis":
                self.axis = attr.i

    def to_function(self, env: Dict[str, NDArray[Any]], constants: Dict[str, NDArray[Any]]) -> List[Function]:
        [v0, indices] = self.node.input
        [v1] = self.node.output
        v0_shape = env[v0].shape

        if indices in constants:
            indices_v = constants[indices]
            if indices_v.dtype == np.int64:
                inode_shape = list(v0_shape)
                inode_shape[self.axis] = 1
                inode_v = np.zeros(inode_shape).astype(np.float32)
                inode_name = gen_unregisterd_node_name(env)
                register_node(env, inode_name, inode_v)

                indices_v = v0_shape[self.axis] + indices_v if indices_v < 0 else indices_v
                starts = [0] * len(v0_shape)
                starts[self.axis] = indices_v.item()
                ends = list(v0_shape)
                ends[self.axis] = indices_v.item() + 1

                return [
                    Slice([v0], [inode_name], starts=starts, ends=ends),  # type: ignore
                    Reshape([inode_name], list(self.node.output), shape=list(map(int, env[v1].shape))),  # type: ignore
                ]
            else:
                raise UnsupportedONNXOperation(self.node, "indices must be integer")
        else:
            raise UnsupportedONNXOperation(self.node, "indices must be constant")
