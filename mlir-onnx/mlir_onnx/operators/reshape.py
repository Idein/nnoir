from mlir.functions import *
from .utils import *


class OpReshape(Op):

    def __init__(self, node):
        super(OpReshape, self).__init__(node)

    def get_dummy_output(self, env):
        [x, shape] = self.node.input
        return env[x].reshape(list(env[shape]))

    def to_function(self, env, constants):
        [x, _] = self.node.input
        [y] = self.node.output
        return [
            Reshape(
                [x],
                list(self.node.output),
                shape=list(map(int, env[y].shape))
            )
        ]
