from typing import Any, List, Set

from .function import Function


class SoftmaxCrossEntropy(Function):
    def __init__(self, inputs: List[bytes], outputs: List[bytes], **params: Any):
        required_params = {"normalize", "cache_score"}
        optional_params: Set[str] = set()
        super(SoftmaxCrossEntropy, self).__init__(inputs, outputs, params, required_params, optional_params)
