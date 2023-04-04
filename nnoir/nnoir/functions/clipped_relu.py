from typing import Any, List, Set

import numpy as np

from .function import Function


class ClippedReLU(Function):
    def __init__(self, inputs: List[bytes], outputs: List[bytes], **params: Any):
        required_params = {"upper"}
        optional_params: Set[str] = set()
        super(ClippedReLU, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, x):  # type: ignore
        return np.maximum(0, np.minimum(x, self.params["upper"]))
