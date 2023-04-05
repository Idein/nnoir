from typing import Any, Dict, Optional, Tuple

from numpy.typing import DTypeLike, NDArray


class Value:
    def __init__(
        self,
        name: bytes,
        np_array: Optional[NDArray[Any]] = None,
        dtype: Optional[bytes] = None,
        shape: Optional[Tuple[int, ...]] = None,
    ):
        self.name = name
        if np_array is None:
            self.dtype = dtype
            self.shape = shape
        else:
            self.dtype = np_array.dtype.str.encode()
            self.shape = np_array.shape

    def dump(self) -> Dict[bytes, Any]:
        return {b"name": self.name, b"dtype": self.dtype, b"shape": self.shape}
