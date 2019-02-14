import msgpack
import numpy
import six
from .mlir import MLIR
from .functions import *
from .value import Value


def load(mlir_file):
    with open(mlir_file, 'rb') as f:
        mlir = msgpack.unpackb(f.read())
    name = mlir[b'mlir'][b'model'][b'name']
    generator_name = mlir[b'mlir'][b'model'][b'generator'][b'name']
    generator_version = mlir[b'mlir'][b'model'][b'generator'][b'version']
    inputs = mlir[b'mlir'][b'model'][b'inputs']
    outputs = mlir[b'mlir'][b'model'][b'outputs']
    vs = [_decode_value(v) for v in mlir[b'mlir'][b'model'][b'values']]
    fs = [_decode_function(f) for f in mlir[b'mlir'][b'model'][b'functions']]
    return MLIR(name, generator_name, generator_version, inputs, outputs, vs, fs)


def _decode_function(function):
    inputs = function[b'inputs']
    outputs = function[b'outputs']
    params = {}
    for k, v in function[b'params'].items():
        if type(v) is dict:
            params[k.decode()] = numpy.load(six.BytesIO(v[b'ndarray']))
        else:
            params[k.decode()] = v
    name = function[b'name'].decode()
    return globals()[name](inputs, outputs, **params)


def _decode_value(value):
    return Value(value[b'name'], value[b'dtype'], value[b'shape'])
