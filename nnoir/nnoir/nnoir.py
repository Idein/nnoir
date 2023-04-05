import re
import typing
from typing import Any, Dict, Iterable, List, Tuple

import msgpack
from numpy.typing import NDArray

from .functions import Function
from .value import Value


class InvalidNNOIRData(Exception):
    def __init__(self, message: str):
        self.message = message


class NNOIR:
    def __init__(
        self,
        name: bytes,
        generator_name: bytes,
        generator_version: bytes,
        inputs: List[bytes],
        outputs: List[bytes],
        values: List[Value],
        functions: List[Function],
        description: bytes = b"",
    ):
        cident = re.compile(rb"^[_A-Za-z][_0-9A-Za-z]*$")
        c_keywords = [
            "auto",
            "break",
            "case",
            "char",
            "const",
            "continue",
            "default",
            "do",
            "double",
            "else",
            "enum",
            "extern",
            "float",
            "for",
            "goto",
            "if",
            "inline",
            "int",
            "long",
            "register",
            "restrict",
            "return",
            "short",
            "signed",
            "sizeof",
            "static",
            "struct",
            "switch",
            "typedef",
            "union",
            "unsigned",
            "void",
            "volatile",
            "while",
            "_Alignas",
            "_Alignof",
            "_Atomic",
            "_Bool",
            "_Complex",
            "_Generic",
            "_Imaginary",
            "_Noreturn",
            "_Static_assert",
            "_Thread_local",
        ]
        if not cident.match(name) or name.decode() in c_keywords:
            raise InvalidNNOIRData(f'graph name "{name.decode()}" MUST be C identifier.')
        self.name = name
        self.description = description
        self.generator_name = generator_name
        self.generator_version = generator_version
        self.inputs = inputs
        self.outputs = outputs
        self.functions = functions
        self.values = values
        vident = re.compile(rb"v[_0-9A-Za-z]*")  # 'v' prefixed
        for v in self.values:
            if not vident.match(v.name) or v.name.decode() in c_keywords:
                raise InvalidNNOIRData(f'value name "{v.name.decode()}" MUST be "v" prefixed C identifier.')

    def to_model(self) -> Dict[bytes, Any]:
        return {
            b"name": self.name,
            b"description": self.description,
            b"generator": {
                b"name": self.generator_name,
                b"version": self.generator_version,
            },
            b"inputs": self.inputs,
            b"outputs": self.outputs,
            b"values": [v.dump() for v in self.values],
            b"functions": [f.dump() for f in self.functions],
        }

    def to_nnoir(self) -> Dict[bytes, Any]:
        return {b"nnoir": {b"version": 0, b"model": self.to_model()}}

    def pack(self) -> bytes:
        return msgpack.packb(self.to_nnoir(), use_bin_type=False)  # type: ignore

    def dump(self, file_name: str) -> None:
        result = self.pack()
        with open(file_name, "w") as f:
            f.buffer.write(result)

    def run(self, *inputs: NDArray[Any]) -> List[NDArray[Any]]:
        env = dict(zip(self.inputs, inputs))

        def cast(vs: Iterable[Tuple[bytes, NDArray[Any]]]) -> List[Tuple[bytes, NDArray[Any]]]:
            res: List[Tuple[bytes, NDArray[Any]]] = []
            for (n, arr) in vs:
                v = [v for v in self.values if v.name == n][0]
                res.append((n, arr.astype(v.dtype)))  # type: ignore
            return res

        def eval(n: bytes) -> NDArray[Any]:
            if n not in env:
                for f in self.functions:
                    if n not in f.outputs:
                        continue
                    inputs = [eval(i) for i in f.inputs]
                    outputs = f.run(*inputs)  # type: ignore
                    if type(outputs) == list:
                        pass
                    elif type(outputs) == tuple:
                        outputs = list(outputs)
                    else:
                        outputs = [typing.cast(NDArray[Any], outputs)]
                    env.update(dict(cast(zip(f.outputs, outputs))))
                    break
            return env[n]

        result = [eval(o) for o in self.outputs]
        del cast
        del eval
        return result
