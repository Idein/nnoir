from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
from nnoir.functions import Function, Slice
from numpy.typing import NDArray

from .utils import *


class OpSlice(Op):
    def __init__(self, node: onnx.NodeProto, *args: Any):
        super(OpSlice, self).__init__(node, *args)

        if self.opset_version < 10:
            raise UnsupportedONNXOperation(self.node, "only opset_version >= 10 is supported")

    def to_function(self, env: Dict[str, NDArray[Any]], constants: Dict[str, NDArray[Any]]) -> List[Function]:
        axes_key = None
        steps_key = None

        if len(self.node.input) == 3:
            [x, starts_key, ends_key] = self.node.input
        if len(self.node.input) == 4:
            [x, starts_key, ends_key, axes_key] = self.node.input
        if len(self.node.input) == 5:
            [x, starts_key, ends_key, axes_key, steps_key] = self.node.input

        x_ndim = env[x].ndim

        if starts_key not in constants:
            raise UnsupportedONNXOperation(self.node, "starts must be constant")
        if ends_key not in constants:
            raise UnsupportedONNXOperation(self.node, "ends must be constant")

        if axes_key is None:
            axes_value = np.arange(x_ndim).astype(np.int32)
        elif axes_key not in constants:
            raise UnsupportedONNXOperation(self.node, "axes must be constant")
        else:
            axes_value = constants[axes_key]

        if steps_key is not None and np.any(np.array(constants[steps_key]) != 1):
            raise UnsupportedONNXOperation(self.node, "# of steps must be None or array of 1")

        # Convert parameter values ​​to values ​​suitable for nnoir.
        x_shape = np.array(env[x].shape)
        axes = np.zeros(x_ndim).astype(np.int32)
        axes[: (max(axes_value) + 1)] = np.bincount(axes_value)
        starts = np.zeros(x_ndim).astype(np.int32)
        starts[axes > 0] = constants[starts_key]
        starts[starts < 0] = x_shape[starts < 0] + x_shape[starts < 0]
        ends = np.array(x_shape)
        ends[axes > 0] = constants[ends_key]
        ends[ends < 0] = x_shape[ends < 0] + ends[ends < 0]
        ends[ends > x_shape] = x_shape[ends > x_shape]

        return [Slice([x], list(self.node.output), starts=starts.tolist(), ends=ends.tolist())]
