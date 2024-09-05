from typing import Any, Dict, List

import numpy as np
import onnx
from nnoir.functions import Transpose
from numpy.typing import NDArray

from .utils import *


class OpWhere(Op):
    def __init__(self, node: onnx.NodeProto, *args: Any):
        super(OpWhere, self).__init__(node, *args)

    def to_function(self, env: Dict[str, NDArray[Any]], constants: Dict[str, NDArray[Any]]) -> List[Function]:
        [condition, x, y] = self.node.input
        if condition in constants:
            condition_value = constants[condition]
            if np.array(condition_value).all() and y in constants:
                return [Transpose([x], list(self.node.output), axes=[i for i in range(env[x].ndim)])]
            elif not np.array(condition_value).any() and x in constants:
                return [Transpose([y], list(self.node.output), axes=[i for i in range(env[y].ndim)])]
            else:
                raise UnsupportedONNXOperation(
                    self.node, "condition value must be all true or all false, and the one not selected must be constant."
                )
        else:
            raise UnsupportedONNXOperation(self.node, "condition must be constant.")
