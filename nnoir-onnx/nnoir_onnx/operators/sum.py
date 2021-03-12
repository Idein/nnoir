import functools
from nnoir.functions import *
from .utils import *


class OpSum(Op):

    def __init__(self, node, *args):
        super(OpSum, self).__init__(node, *args)

    def to_function(self, env, constants):
        if len(self.node.input) != 2:
            raise UnsupportedONNXOperation(self.node, '# of inputs must be 2')
        return [Add(list(self.node.input), list(self.node.output))]
