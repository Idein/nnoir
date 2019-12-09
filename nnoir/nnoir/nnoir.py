import msgpack
import re


class InvalidNNOIRData(Exception):

    def __init__(self, message):
        self.message = message


class NNOIR():

    def __init__(self, name, generator_name, generator_version, inputs, outputs, values, functions):
        cident = re.compile(rb'^[_A-Za-z][_0-9A-Za-z]*$')
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
            raise InvalidNNOIRData('graph name "{}" MUST be C identifier.'.format(name))
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
                raise InvalidNNOIRData('value name "{}" MUST be "v" prefixed C identifier.'.format(v.name))

    def to_model(self):
        return {
            b'name': self.name,
            b'generator':
            {b'name': self.generator_name,
             b'version': self.generator_version},
            b'inputs': self.inputs,
            b'outputs': self.outputs,
            b'values': [v.dump() for v in self.values],
            b'functions': [f.dump() for f in self.functions]
        }

    def to_nnoir(self):
        return {
            b'nnoir':
            {b'version': 0,
             b'model': self.to_model()
             }
        }

    def pack(self):
        return msgpack.packb(self.to_nnoir())

    def dump(self, file_name):
        result = self.pack()
        with open(file_name, 'w') as f:
            f.buffer.write(result)

    def run(self, *inputs):
        env = dict(zip(self.inputs, inputs))

        def eval(n):
            if n not in env:
                for f in self.functions:
                    if n not in f.outputs:
                        continue
                    inputs = [eval(i) for i in f.inputs]
                    outputs = f.run(*inputs)
                    if type(outputs) == list:
                        pass
                    elif type(outputs) == tuple:
                        outputs = list(outputs)
                    else:
                        outputs = [outputs]
                    env.update(dict(zip(f.outputs, outputs)))
                    break
            return env[n]

        return [eval(o) for o in self.outputs]
