import msgpack
import chainer
import sys
from .converter import *
from mlir.load import load


class MLIRFunction(chainer.Chain):
    def __init__(self, mlir_file):
        super(MLIRFunction, self).__init__()
        self.mlir = load(mlir_file)
        self._init_functions()

    def _init_functions(self):
        self.functions = {}
        nodes = {n.name: n for n in self.mlir.values}
        for function in self.mlir.functions:
            inputs = [nodes[x] for x in function.inputs]
            outputs = [nodes[x] for x in function.outputs]
            self.functions[id(function)] = globals()["Convert" + function.__class__.__name__](function, inputs, outputs)

    def __call__(self, *xs):
        if len(self.mlir.inputs) != len(xs):
            raise Exception("the number of input variables and the expected number of inputs do not match.")
        variable_dict = {inp: x for inp, x in zip(self.mlir.inputs, xs)}
        done = set()
        while len(done) < len(self.mlir.functions):
            found = True
            for function in self.mlir.functions:
                if id(function) not in done and set(function.inputs).issubset(variable_dict.keys()):
                    variable_dict[function.outputs[0]] = self.functions[id(function)](
                        *[variable_dict[x] for x in function.inputs])
                    done.add(id(function))
                    found = True
                    break
            if not found:
                raise Exception("invalid mlir.")
        ret = tuple(variable_dict[x] for x in self.mlir.outputs)
        if len(ret) == 1:
            return ret[0]
        else:
            return ret
