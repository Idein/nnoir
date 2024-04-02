from typing import Any, Dict, List, Optional, Tuple
from functools import reduce

import numpy as np
import onnx
from nnoir.functions import Function, Slice
from numpy.typing import NDArray

from .utils import *


class OpSlice(Op):
    def __init__(self, node: onnx.NodeProto, *args: Any):
        super(OpSlice, self).__init__(node, *args)

    def to_function(self, env: Dict[str, NDArray[Any]], constants: Dict[str, NDArray[Any]]) -> List[Function]:
        axes  = None
        steps = None

        if len(self.node.input) == 3:
            [x, starts, ends] = self.node.input
        if len(self.node.input) == 4:
            [x, starts, ends, axes] = self.node.input
        if len(self.node.input) == 5:
            [x, starts, ends, axes, steps] = self.node.input

        if starts not in constants:
            raise UnsupportedONNXOperation(self.node, "starts must be constant")
        if ends not in constants:
            raise UnsupportedONNXOperation(self.node, "ends must be constant")

        if axes is None:
            axes_v = np.arange(env[x].ndim).astype(np.int32)
        elif axes not in constants:
            raise UnsupportedONNXOperation(self.node, "axes must be constant")
        else:
            axes_v = constants[axes]
        
                
        if steps is not None and not np.all(np.array(steps) == 1):
            UnsupportedONNXOperation(self.node, "# of steps must be None or array of 1")
        
        return [Slice([x], list(self.node.output), starts=constants[starts], ends=constants[ends], axes=axes_v)]


