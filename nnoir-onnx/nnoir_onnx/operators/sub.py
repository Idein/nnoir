from nnoir.functions import *
from .utils import *


class OpSub(Op):

    def __init__(self, node, *args):
        super(OpSub, self).__init__(node, *args)

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
