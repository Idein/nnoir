from typing import Any, Dict, List, Optional, Tuple

import onnx
from nnoir.functions import Function, Transpose
from numpy.typing import NDArray

from .utils import *


class OpDropout(Op):
    def __init__(self, node: onnx.NodeProto, *args: Any):
        super(OpDropout, self).__init__(node, *args)

    def to_function(self, env: Dict[str, NDArray[Any]], constants: Dict[str, NDArray[Any]]) -> List[Function]:
        return [
            Transpose(
                list(self.node.input),
                list(self.node.output[:1]),
                axes=list(range(len(env[self.node.input[0]].shape))),
            )
        ]
