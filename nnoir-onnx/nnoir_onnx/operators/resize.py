from nnoir.functions import *
from .utils import *


class OpResize(Op):

    def __init__(self, node):
        super(OpResize, self).__init__(node)

    def to_function(self, env, constants):
        [x, _] = self.node.input
        [y] = self.node.output
        return [
            Resize(
                [x],
                list(self.node.output),
                size=tuple(env[y].shape[2:]),
                mode='align_none'
            )
        ]
