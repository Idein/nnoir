import msgpack
import six
import re


class InvalidMLIRData(Exception):

    def __init__(self, message):
        self.message = message


class MLIR():

    def __init__(self, name, generator_name, generator_version, inputs, outputs, values, functions):
        cident = re.compile(rb'[_A-Za-z][_0-9A-Za-z]*')
        c_keywords = [
            'auto', 'break', 'case',
            'char', 'const', 'continue',
            'default', 'do', 'double',
            'else', 'enum', 'extern',
            'float', 'for', 'goto',
            'if', 'inline', 'int',
            'long', 'register', 'restrict',
            'return', 'short', 'signed',
            'sizeof', 'static', 'struct',
            'switch', 'typedef', 'union',
            'unsigned', 'void', 'volatile',
            'while', '_Alignas', '_Alignof',
            '_Atomic', '_Bool', '_Complex',
            '_Generic', '_Imaginary', '_Noreturn',
            '_Static_assert', '_Thread_local'
        ]
        if not cident.match(name) or name.decode() in c_keywords:
            raise InvalidMLIRData('graph name "{}" MUST be C identifier.'.format(name))
        self.name = name
        self.generator_name = generator_name
        self.generator_version = generator_version
        self.inputs = inputs
        self.outputs = outputs
        self.functions = functions
        self.values = values
        vident = re.compile(rb'v[_0-9A-Za-z]*')  # 'v' prefixed
        for v in self.values:
            if not vident.match(v.name) or v.name.decode() in c_keywords:
                raise InvalidMLIRData('value name "{}" MUST be "v" prefixed C identifier.'.format(v.name))

    def pack(self):
        return msgpack.packb(
            {b'mlir':
             {b'version': 0,
              b'model':
              {b'name': self.name,
               b'generator':
               {b'name': self.generator_name,
                b'version': self.generator_version},
               b'inputs': self.inputs,
               b'outputs': self.outputs,
                  b'values': [n.dump() for n in self.values],
                  b'functions': [e.dump() for e in self.functions]
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
