from typing import Any, List, Set

import numpy as np

from .function import Function


class Gemm(Function):
    def __init__(self, inputs: List[bytes], outputs: List[bytes], **params: Any):
        required_params = {"transA", "transB", "alpha", "beta"}
        optional_params: Set[str] = set()
        super(Gemm, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, a, b, c=None):  # type: ignore
        a = a if not self.params["transA"] else a.T
        b = b if not self.params["transB"] else b.T
        result = self.params["alpha"] * np.dot(a, b)  # type: ignore
        if c is not None:
            result += self.params["beta"] * c
        return result
