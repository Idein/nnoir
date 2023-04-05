from typing import Any, List, Set

from .function import Function


class Sub(Function):
    def __init__(self, inputs: List[bytes], outputs: List[bytes], **params: Any):
        required_params: Set[str] = set()
        optional_params: Set[str] = set()
        super(Sub, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, x1, x2):  # type: ignore
        return x1 - x2
