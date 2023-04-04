from typing import Any, List, Set

import numpy as np

from .function import Function


class MatMul(Function):
    def __init__(self, inputs: List[bytes], outputs: List[bytes], **params: Any):
        required_params: Set[str] = set()
        optional_params: Set[str] = set()
        super(MatMul, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, a, b):  # type: ignore
        return np.matmul(a, b)
