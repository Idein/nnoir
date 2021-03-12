from nnoir.functions import *
from .utils import *


class OpRelu(Op):

    def __init__(self, node, *args):
        super(OpRelu, self).__init__(node, *args)

    def to_function(self, env, constants):
        return [ReLU(list(self.node.input), list(self.node.output))]
