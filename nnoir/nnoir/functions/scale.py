from typing import Any, List, Set

import numpy as np

from .function import Function


class Scale(Function):
    def __init__(self, inputs: List[bytes], outputs: List[bytes], **params: Any):
        required_params = {"axis", "W", "b"}
        optional_params: Set[str] = set()
        super(Scale, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, x):  # type: ignore
        shape_post_len = len(x.shape) - self.params["axis"] - len(self.params["W"].shape)
        shape = (1,) * self.params["axis"] + self.params["W"].shape + (1,) * shape_post_len
        result = x * np.reshape(self.params["W"], shape)
        if self.params["b"] is not None:
            result += np.reshape(self.params["b"], shape)
        return result
