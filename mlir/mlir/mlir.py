import msgpack
import six
import re

class InvalidMLIRData(Exception):

    def __init__(self, message):
        self.message = message

class MLIR():

    def __init__(self, name, generator_name, generator_version, inputs, outputs, nodes, edges):
        cident = re.compile(rb'[_A-Za-z][_0-9A-Za-z]*') # TODO: remove C keywords
        if not cident.match(name):
            raise InvalidMLIRData('graph name "{}" MUST be C identifier.'.format(name))
        self.name = name
        self.generator_name = generator_name
        self.generator_version = generator_version
        self.inputs = inputs
        self.outputs = outputs
        self.edges = edges
        self.nodes = nodes
        for n in self.nodes:
            if not cident.match(n.name):
                raise InvalidMLIRData('node name "{}" MUST be C identifier.'.format(n.name))

    def pack(self):
        return msgpack.packb(
            { b'mlir':
              { b'version' : 0,
                b'model' :
                { b'name': self.name,
                  b'generator':
                { b'name' : self.generator_name,
                  b'version': self.generator_version },
                  b'inputs': self.inputs,
                  b'outputs': self.outputs,
                  b'nodes': [ n.dump() for n in self.nodes],
                  b'edges': [ e.dump() for e in self.edges]
                }
              }
            }
        )

    def dump(self, file_name):
        result = self.pack()
        with open(file_name, 'w') as f:
            if six.PY2:
                six.print_(result, file=f)
            else:
                f.buffer.write(result)
