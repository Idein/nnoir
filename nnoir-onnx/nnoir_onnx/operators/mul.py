from nnoir.functions import *
from .utils import *

class OpMul(Op):

    def __init__(self, node):
        super(OpMul, self).__init__(node)

    def get_dummy_output(self, env):
        [a, b] = self.node.input
        return env[a] * env[b]

    def to_function(self, env, constants):
        [a, b] = self.node.input
        if a in constants and b not in constants:
            return [MulConstant([b], list(self.node.output), value=encode_ndarray(constants[a]))]
        elif a not in constants and b in constants:
            return [MulConstant([a], list(self.node.output), value=encode_ndarray(constants[b]))]
        elif a not in constants and b not in constants:
            return [Mul(list(self.node.input), list(self.node.output))]
        else:
            raise UnsupportedONNXOperation(self.node, 'bug! (unreachable here)')
