from typing import Any, List, Set

from .function import Function


class Reshape(Function):
    def __init__(self, inputs: List[bytes], outputs: List[bytes], **params: Any):
        required_params = {"shape"}
        optional_params: Set[str] = set()
        super(Reshape, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, x):  # type: ignore
        return x.reshape(self.params["shape"])
