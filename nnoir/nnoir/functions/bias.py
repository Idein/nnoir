from typing import Any, List, Set

import numpy as np

from .function import Function


class Bias(Function):
    def __init__(self, inputs: List[bytes], outputs: List[bytes], **params: Any):
        required_params = {"axis", "b"}
        optional_params: Set[str] = set()
        super(Bias, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, x):  # type: ignore
        shape_post_len = len(x.shape) - self.params["axis"] - len(self.params["b"].shape)
        shape = (1,) * self.params["axis"] + self.params["b"].shape + (1,) * shape_post_len
        return x + np.reshape(self.params["b"], shape)
