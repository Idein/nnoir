from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
from nnoir.functions import ClippedReLU, Function
from numpy.typing import NDArray

from .utils import *


class OpClip(Op):
    def __init__(self, node: onnx.NodeProto, *args: Any):
        super(OpClip, self).__init__(node, *args)

    def to_function(self, env: Dict[str, NDArray[Any]], constants: Dict[str, NDArray[Any]]) -> List[Function]:
        _min: Union[float, NDArray[Any]] = -3.4028234663852886e38
        _max: Union[float, NDArray[Any]] = 3.4028234663852886e38

        if self.opset_version < 6:
            raise UnsupportedONNXOperation(self.node, "only opset_version >= 6 is supported")

        if self.opset_version >= 11:

            if len(self.node.input) >= 2:
                _min = constants[self.node.input[1]]
                if isinstance(_min, np.ndarray):
                    _min = float(_min.ravel()[0])

            if len(self.node.input) >= 3:
                _max = constants[self.node.input[2]]
                if isinstance(_max, np.ndarray):
                    _max = float(_max.ravel()[0])

            if _min != 0.0:
                raise UnsupportedONNXOperation(self.node, "min must be 0.0")

            return [ClippedReLU([self.node.input[0]], list(self.node.output), upper=_max)]

        else:
            # opset_version 6
            for attr in self.node.attribute:
                if attr.name == "max":
                    _max = attr.f
                if attr.name == "min":
                    _min = attr.f

            if _min != 0.0:
                raise UnsupportedONNXOperation(self.node, "min must be 0.0")

            return [ClippedReLU(list(self.node.input), list(self.node.output), upper=_max)]
