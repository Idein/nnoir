import msgpack
import six

class MLIR():
    def __init__(self, name, generator_name, generator_version, inputs, outputs, nodes, edges):
        self.name = name
        self.generator_name = generator_name
        self.generator_version = generator_version
        self.inputs = inputs
        self.outputs = outputs
        self.edges = edges
        self.nodes = nodes
    def dump(self, file_name):
        result = { b'mlir':
                   { b'version' : 0,
                     b'model' :
                     { b'name': self.name,
                       b'generator':
                       { b'name' : self.generator_name,
                         b'version': self.generator_version },
                       b'inputs': self.inputs,
                       b'outputs': self.outputs,
                       b'nodes': [ n.dump() for n in self.nodes],
                       b'edges': [ e.dump() for e in self.edges] } } }
        with open(file_name, 'w') as f:
            if six.PY2:
                six.print_(msgpack.packb(result), file=f)
            else:
                f.buffer.write(msgpack.packb(result))
