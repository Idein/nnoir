from typing import Any, List, Set

import numpy as np

from .function import Function


class Concat(Function):
    def __init__(self, inputs: List[bytes], outputs: List[bytes], **params: Any):
        required_params = {"axis"}
        optional_params: Set[str] = set()
        super(Concat, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, *xs):  # type: ignore
        return np.concatenate(xs, self.params["axis"])  # type: ignore
