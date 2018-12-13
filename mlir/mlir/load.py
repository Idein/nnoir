import msgpack
import numpy
import six
from .mlir import MLIR
from .edges import *
from .node import Node

def load(mlir_file):
    with open(mlir_file, 'rb') as f:
        mlir = msgpack.unpackb(f.read())
    name = mlir[b'mlir'][b'model'][b'name']
    generator_name = mlir[b'mlir'][b'model'][b'generator'][b'name']
    generator_version = mlir[b'mlir'][b'model'][b'generator'][b'version']
    inputs = mlir[b'mlir'][b'model'][b'inputs']
    outputs = mlir[b'mlir'][b'model'][b'outputs']
    nodes = [ _decode_node(n) for n in mlir[b'mlir'][b'model'][b'nodes'] ]
    edges = [ _decode_edge(e) for e in mlir[b'mlir'][b'model'][b'edges'] ]
    return MLIR(name, generator_name, generator_version, inputs, outputs, nodes, edges)

def _decode_edge(edge):
    inputs = edge[b'inputs']
    outputs = edge[b'outputs']
    params = {}
    for k,v in edge[b'params'].items():
        if type(v) is dict:
            params[k.decode()] = numpy.load(six.BytesIO(v[b'ndarray']))
        else:
            params[k.decode()] = v
    name = edge[b'name'].decode()
    return globals()[name](inputs, outputs, **params)

def _decode_node(node):
    return Node(node[b'name'], node[b'dtype'], node[b'shape'])
