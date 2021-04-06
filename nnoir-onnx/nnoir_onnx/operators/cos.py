from nnoir.functions import *

from .utils import *


class OpCos(Op):
    def __init__(self, node, *args):
        super(OpCos, self).__init__(node, *args)

    def to_function(self, env, constants):
        return [Cos(list(self.node.input), list(self.node.output))]
