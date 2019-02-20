import numpy as np
from nnoir.functions import *
from .utils import *


class OpTranspose(Op):

    def __init__(self, node):
        super(OpTranspose, self).__init__(node)

        self.perm = None
        for attr in node.attribute:
            if attr.name == 'perm':
                self.perm = list(attr.ints)

    def get_dummy_output(self, env):
        [x] = self.node.input
        return np.transpose(env[x], self.perm)

    def to_function(self, env, constants):
        return [
            Transpose(
                list(self.node.input),
                list(self.node.output),
                axes=self.perm
            )
        ]
