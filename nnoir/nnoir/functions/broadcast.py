from typing import Any, List, Set

import numpy as np

from .function import Function


class BroadcastTo(Function):
    def __init__(self, inputs: List[bytes], outputs: List[bytes], **params: Any):
        required_params = {"shape"}
        optional_params: Set[str] = set()
        super(BroadcastTo, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, x):  # type: ignore
        return np.broadcast_to(x, self.params["shape"])  # type: ignore
