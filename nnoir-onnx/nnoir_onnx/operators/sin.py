from nnoir.functions import *

from .utils import *


class OpSin(Op):
    def __init__(self, node, *args):
        super(OpSin, self).__init__(node, *args)

    def to_function(self, env, constants):
        return [Sin(list(self.node.input), list(self.node.output))]
