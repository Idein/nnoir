import io
import msgpack
import numpy
from .nnoir import NNOIR
from .functions import *
from .value import Value


def load(nnoir_file):
    with open(nnoir_file, 'rb') as f:
        nnoir = msgpack.unpackb(f.read())
    name = nnoir[b'nnoir'][b'model'][b'name']
    generator_name = nnoir[b'nnoir'][b'model'][b'generator'][b'name']
    generator_version = nnoir[b'nnoir'][b'model'][b'generator'][b'version']
    inputs = nnoir[b'nnoir'][b'model'][b'inputs']
    outputs = nnoir[b'nnoir'][b'model'][b'outputs']
    vs = [_decode_value(v) for v in nnoir[b'nnoir'][b'model'][b'values']]
    fs = [_decode_function(f) for f in nnoir[b'nnoir'][b'model'][b'functions']]
    return NNOIR(name, generator_name, generator_version, inputs, outputs, vs, fs)


def _decode_function(function):
    inputs = function[b'inputs']
    outputs = function[b'outputs']
    params = {}
    for k, v in function[b'params'].items():
        if type(v) is dict:
            params[k.decode()] = numpy.load(io.BytesIO(v[b'ndarray']))
        else:
            params[k.decode()] = v
    name = function[b'name'].decode()
    return globals()[name](inputs, outputs, **params)


def _decode_value(value):
    return Value(value[b'name'], np_array=None, dtype=value[b'dtype'], shape=value[b'shape'])
