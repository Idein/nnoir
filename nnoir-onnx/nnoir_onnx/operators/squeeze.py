from .utils import *
from nnoir.functions import *

import numpy as np


class OpSqueeze(Op):

    def __init__(self, node, *args):
        super(OpSqueeze, self).__init__(node, *args)

        self.axes = []

        for attr in node.attribute:
            if attr.name == 'axes':
                self.axes = attr.ints

    def to_function(self, env, constants):
        [x] = self.node.input
        [y] = self.node.output
        shape0 = env[x].shape

        if self.axes == []:
            shape1 = [sh for sh in shape0 if sh != 1]
        else:
            sh = list(shape0)
            dim = len(sh)
            for a in sorted([(a + dim) % dim for a in self.axes], reverse=True):
                del sh[a]
            shape1 = sh

        return [Reshape([x], [y], shape=tuple(shape1))]
