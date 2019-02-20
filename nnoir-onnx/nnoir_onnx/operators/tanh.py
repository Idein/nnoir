from nnoir.functions import *
from .utils import *


class OpTanh(Op):

    def __init__(self, node):
        super(OpTanh, self).__init__(node)

    def get_dummy_output(self, env):
        [x] = self.node.input
        return env[x]

    def to_function(self, env, constants):
        return [Tanh(list(self.node.input), list(self.node.output))]
