from typing import Any, List, Set

from .function import Function


class AddConstant(Function):
    def __init__(self, inputs: List[bytes], outputs: List[bytes], **params: Any):
        required_params = {"value"}
        optional_params: Set[str] = set()
        super(AddConstant, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, x):  # type: ignore
        return x + self.params["value"]
