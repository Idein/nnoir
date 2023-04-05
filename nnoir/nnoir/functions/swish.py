from typing import Any, List, Set

import numpy as np

from .function import Function


class Swish(Function):
    def __init__(self, inputs: List[bytes], outputs: List[bytes], **params: Any):
        required_params = {"beta"}
        optional_params: Set[str] = set()
        super(Swish, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, x):  # type: ignore
        beta = self.params["beta"]
        d = len(x.shape) - 1 - len(beta.shape)
        beta = beta.reshape((1,) + beta.shape + (1,) * d)
        return x * (0.5 * np.tanh(0.5 * beta * x) + 0.5)
