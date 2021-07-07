from nnoir.functions import *

from .utils import *


class OpExp(Op):
    def __init__(self, node, *args):
        super(OpExp, self).__init__(node, *args)

    def to_function(self, env, constants):
        return [Exp(list(self.node.input), list(self.node.output))]
