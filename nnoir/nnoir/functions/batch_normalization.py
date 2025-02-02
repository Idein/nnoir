from typing import Any, List, Set

import numpy as np

from .function import Function


class BatchNormalization(Function):
    def __init__(self, inputs: List[bytes], outputs: List[bytes], **params: Any):
        required_params = {"eps", "avg_mean", "avg_var", "gamma", "beta"}
        optional_params: Set[str] = set()
        super(BatchNormalization, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, x):  # type: ignore
        if self.params["gamma"] is None:
            self.params["gamma"] = np.ones(self.params["avg_mean"].size).astype(np.float32)
        if self.params["beta"] is None:
            self.params["beta"] = np.zeros(self.params["avg_mean"].size).astype(np.float32)

        shape = (1, self.params["gamma"].size) + ((1,) * (len(x.shape) - 2))  # type: ignore
        gamma = self.params["gamma"].reshape(shape)  # type: ignore
        beta = self.params["beta"].reshape(shape)  # type: ignore
        avg_mean = self.params["avg_mean"].reshape(shape)
        avg_var = self.params["avg_var"].reshape(shape)
        return gamma * (x - avg_mean) / np.sqrt(avg_var + self.params["eps"]) + beta
