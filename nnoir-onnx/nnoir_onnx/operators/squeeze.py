import numpy as np
from nnoir.functions import *

from .utils import *


class OpSqueeze(Op):
    def __init__(self, node, *args):
        super(OpSqueeze, self).__init__(node, *args)

        self.axes = []

        for attr in node.attribute:
            if attr.name == "axes":
                self.axes = attr.ints

    def to_function(self, env, constants):
        x = self.node.input[0]
        [y] = self.node.output
        shape0 = env[x].shape

        axes = []
        if self.axes != []:
            axes = self.axes
        if len(self.node.input) > 1:  # Opset 13
            axes = list(env[self.node.input[1]])

        if axes == []:
            shape1 = [sh for sh in shape0 if sh != 1]
        else:
            sh = list(shape0)
            dim = len(sh)
            for a in sorted([(a + dim) % dim for a in axes], reverse=True):
                del sh[a]
            shape1 = sh

        return [Reshape([x], [y], shape=tuple(shape1))]
