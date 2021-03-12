import numpy as np
from nnoir.functions import *
from .utils import *


class OpConcat(Op):

    def __init__(self, node, *args):
        super(OpConcat, self).__init__(node, *args)

        self.axis = None
        for attr in node.attribute:
            if attr.name == 'axis':
                self.axis = attr.i

    def to_function(self, env, constants):
        [y] = self.node.output
        axis = len(env[y].shape) + self.axis if self.axis < 0 else self.axis
        return [
            Concat(
                list(self.node.input),
                list(self.node.output),
                axis=int(axis)
            )
        ]
