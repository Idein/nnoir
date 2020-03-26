from nnoir.functions import *
from .utils import *


class OpTanh(Op):

    def __init__(self, node, *args):
        super(OpTanh, self).__init__(node, *args)

    def to_function(self, env, constants):
        return [Tanh(list(self.node.input), list(self.node.output))]
