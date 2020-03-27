from nnoir.functions import *
from .utils import *


class OpMul(Op):

    def __init__(self, node, *args):
        super().__init__(node, *args)

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
