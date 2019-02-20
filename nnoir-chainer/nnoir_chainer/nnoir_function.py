import msgpack
import chainer
import sys
from .converter import *
from nnoir.load import load


class NNOIRFunction(chainer.Chain):
    def __init__(self, nnoir_file):
        super(NNOIRFunction, self).__init__()
        self.nnoir = load(nnoir_file)
        self._init_functions()

    def _init_functions(self):
        self.functions = {}
        nodes = {n.name: n for n in self.nnoir.values}
        for function in self.nnoir.functions:
            inputs = [nodes[x] for x in function.inputs]
            outputs = [nodes[x] for x in function.outputs]
            self.functions[id(function)] = globals()["Convert" + function.__class__.__name__](function, inputs, outputs)

    def __call__(self, *xs):
        if len(self.nnoir.inputs) != len(xs):
            raise Exception("the number of input variables and the expected number of inputs do not match.")
        variable_dict = {inp: x for inp, x in zip(self.nnoir.inputs, xs)}
        done = set()
        while len(done) < len(self.nnoir.functions):
            found = True
            for function in self.nnoir.functions:
                if id(function) not in done and set(function.inputs).issubset(variable_dict.keys()):
                    variable_dict[function.outputs[0]] = self.functions[id(function)](
                        *[variable_dict[x] for x in function.inputs])
                    done.add(id(function))
                    found = True
                    break
            if not found:
                raise Exception("invalid nnoir.")
        ret = tuple(variable_dict[x] for x in self.nnoir.outputs)
        if len(ret) == 1:
            return ret[0]
        else:
            return ret
