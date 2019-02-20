from nnoir.functions import *
from .utils import *


class OpDropout(Op):

    def __init__(self, node):
        super(OpDropout, self).__init__(node)

    def get_dummy_output(self, env):
        [x] = self.node.input
        return env[x]

    def to_function(self, env, constants):
        return [
            Transpose(
                list(self.node.input),
                list(self.node.output[:1]),
                axes=list(range(len(env[self.node.input[0]].shape)))
            )
        ]
