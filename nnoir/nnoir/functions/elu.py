from typing import Any, List, Set

import numpy as np

from .function import Function


class ELU(Function):
    def __init__(self, inputs: List[bytes], outputs: List[bytes], **params: Any):
        required_params = {"alpha"}
        optional_params: Set[str] = set()
        super(ELU, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, x):  # type: ignore
        R = x.copy()
        v = R < 0
        R[v] = self.params["alpha"] * (np.exp(R[v]) - 1)
        return R
