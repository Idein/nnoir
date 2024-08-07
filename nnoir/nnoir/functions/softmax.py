from typing import Any, List, Set

import numpy as np

from .function import Function


class Softmax(Function):
    def __init__(self, inputs: List[bytes], outputs: List[bytes], **params: Any):
        required_params = {"axis"}
        optional_params: Set[str] = set()
        super(Softmax, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, x):  # type: ignore
        return np.exp(x) / np.sum(np.exp(x), self.params["axis"], keepdims=True)
