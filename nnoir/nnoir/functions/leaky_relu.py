from typing import Any, List, Set

from .function import Function


class LeakyReLU(Function):
    def __init__(self, inputs: List[bytes], outputs: List[bytes], **params: Any):
        required_params = {"slope"}
        optional_params: Set[str] = set()
        super(LeakyReLU, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, x):  # type: ignore
        R = x.copy()
        R[R < 0] *= self.params["slope"]
        return R
