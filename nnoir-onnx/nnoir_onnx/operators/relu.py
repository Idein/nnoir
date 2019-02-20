from nnoir.functions import *
from .utils import *


class OpRelu(Op):

    def __init__(self, node):
        super(OpRelu, self).__init__(node)

    def get_dummy_output(self, env):
        [x] = self.node.input
        return env[x]

    def to_function(self, env, constants):
        return [ReLU(list(self.node.input), list(self.node.output))]
