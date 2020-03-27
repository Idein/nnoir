from nnoir.functions import *
from .utils import *


class OpUnsqueeze(Op):

    def __init__(self, node, *args):
        super(OpUnsqueeze, self).__init__(node, *args)

    def to_function(self, env, constants):
        [x] = self.node.input
        [y] = self.node.output

        # The axes attribute is ignored. We already know output shape,
        # without reconstruction from input shape and axes.

        return [
            Reshape(
                [x],
                list(self.node.output),
                shape=list(map(int, env[y].shape))
            )
        ]
