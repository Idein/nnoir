from typing import Any, List, Set

import numpy as np

from .function import Function


class Sqrt(Function):
    def __init__(self, inputs: List[bytes], outputs: List[bytes], **params: Any):
        required_params: Set[str] = set()
        optional_params: Set[str] = set()
        super(Sqrt, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, x):  # type: ignore
        return np.sqrt(x)
