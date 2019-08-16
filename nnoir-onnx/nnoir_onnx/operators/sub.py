from nnoir.functions import *
from .utils import *


class OpSub(Op):

    def __init__(self, node):
        super(OpSub, self).__init__(node)

    def get_dummy_output(self, env):
        [a, b] = self.node.input
        return env[a] - env[b]

    def to_function(self, env, constants):
        [a, b] = self.node.input
        if a in constants and b not in constants:
            raise UnsupportedONNXOperation(self.node, 'unimplemented yet')
        elif a not in constants and b in constants:
            return [Bias([a], list(self.node.output), axis=0, b=encode_ndarray(-constants[b]))]
        elif a not in constants and b not in constants:
            return [Sub(list(self.node.input), list(self.node.output))]
        else:
            raise UnsupportedONNXOperation(self.node, 'bug! (unreachable here)')
