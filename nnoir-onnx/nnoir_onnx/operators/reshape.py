from nnoir.functions import *
from .utils import *


class OpReshape(Op):

    def __init__(self, node):
        super(OpReshape, self).__init__(node)

    def get_dummy_output(self, env):
        [x, shape] = self.node.input
        x = env[x]
        shape = env[shape]
        for i in range(len(shape)):
            if shape[i] == 0:
                shape[i] = x.shape[i]
        return x.reshape(shape)

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
