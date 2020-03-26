from nnoir.functions import *
from .utils import *


class OpSoftmax(Op):

    def __init__(self, node, *args):
        super(OpSoftmax, self).__init__(node, *args)

        self.axis = 1
        for attr in self.node.attribute:
            if attr.name == 'axis':
                self.axis = attr.i

    def to_function(self, env, constants):
        return [
            Softmax(
                list(self.node.input),
                list(self.node.output),
                axis=self.axis
            )
        ]
