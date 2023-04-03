import io
from typing import Any, Dict

import msgpack
import numpy

from .functions import *
from .nnoir import NNOIR
from .value import Value


def load(nnoir_file: str) -> NNOIR:
    with open(nnoir_file, "rb") as f:
        nnoir = msgpack.unpackb(f.read(), raw=True)
    name = nnoir[b"nnoir"][b"model"][b"name"]
    generator_name = nnoir[b"nnoir"][b"model"][b"generator"][b"name"]
    description = b""
    if b"description" in nnoir[b"nnoir"][b"model"]:
        description = nnoir[b"nnoir"][b"model"][b"description"]
    generator_version = nnoir[b"nnoir"][b"model"][b"generator"][b"version"]
    inputs = nnoir[b"nnoir"][b"model"][b"inputs"]
    outputs = nnoir[b"nnoir"][b"model"][b"outputs"]
    vs = [_decode_value(v) for v in nnoir[b"nnoir"][b"model"][b"values"]]
    fs = [_decode_function(f) for f in nnoir[b"nnoir"][b"model"][b"functions"]]
    return NNOIR(name, generator_name, generator_version, inputs, outputs, vs, fs, description)


def _decode_function(function: Dict[bytes, Any]) -> Function:
    inputs = function[b"inputs"]
    outputs = function[b"outputs"]
    params = {}
    for k, v in function[b"params"].items():
        if type(v) is dict:
            params[k.decode()] = numpy.load(io.BytesIO(v[b"ndarray"]))  # type: ignore
        else:
            params[k.decode()] = v
    name = function[b"name"].decode()
    return globals()[name](inputs, outputs, **params)  # type: ignore


def _decode_value(value: Dict[bytes, Any]) -> Value:
    return Value(value[b"name"], np_array=None, dtype=value[b"dtype"], shape=value[b"shape"])
