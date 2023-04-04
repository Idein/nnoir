from typing import Any, Dict, List, Set, Tuple

import numpy as np
from numpy.typing import NDArray

from .function import Function


class Add(Function):
    def __init__(self, inputs: List[bytes], outputs: List[bytes], **params: Any):
        required_params: Set[str] = set()
        optional_params: Set[str] = set()
        super(Add, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, x1, x2):  # type: ignore
        return x1 + x2
