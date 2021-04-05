from nnoir.functions import *

from .utils import *


class OpTan(Op):
    def __init__(self, node, *args):
        super(OpTan, self).__init__(node, *args)

    def to_function(self, env, constants):
        return [Tan(list(self.node.input), list(self.node.output))]
