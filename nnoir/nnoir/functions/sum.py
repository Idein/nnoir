from typing import Any, List, Set

import numpy as np

from . import util
from .function import Function


class Sum(Function):
    def __init__(self, inputs: List[bytes], outputs: List[bytes], **params: Any):
        required_params = {"axes", "keepdims"}
        optional_params: Set[str] = set()
        super(Sum, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, x):  # type: ignore
        return np.sum(x, axis=tuple(self.params["axes"]), keepdims=self.params["keepdims"])
