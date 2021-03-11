from typing import Iterable, TypeVar
from nnoir_onnx import operators


T = TypeVar('T')

NNOIR_SUPPORTED_OPS = {op[2:] for op in operators.__dict__ if op.startswith('Op')}


def index_of(iterable: Iterable[T], element: T) -> int:
    for idx, el in enumerate(iterable):
        if el == element:
            return idx
    return -1
