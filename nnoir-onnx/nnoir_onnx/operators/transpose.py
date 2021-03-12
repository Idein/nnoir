import numpy as np
from nnoir.functions import *
from .utils import *


class OpTranspose(Op):

    def __init__(self, node, *args):
        super(OpTranspose, self).__init__(node, *args)

        self.perm = None
        for attr in node.attribute:
            if attr.name == 'perm':
                self.perm = list(attr.ints)

    def to_function(self, env, constants):
        return [
            Transpose(
                list(self.node.input),
                list(self.node.output),
                axes=self.perm
            )
        ]
