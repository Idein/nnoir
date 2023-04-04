from typing import Any, List, Set

import numpy as np

from .function import Function


class Tanh(Function):
    def __init__(self, inputs: List[bytes], outputs: List[bytes], **params: Any):
        required_params: Set[str] = set()
        optional_params: Set[str] = set()
        super(Tanh, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, x):  # type: ignore
        R = x.copy()
        return np.tanh(R)
