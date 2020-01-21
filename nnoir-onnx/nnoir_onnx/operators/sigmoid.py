from nnoir.functions import *
from .utils import *


class OpSigmoid(Op):

    def __init__(self, node):
        super(OpSigmoid, self).__init__(node)

    def to_function(self, env, constants):
        return [Sigmoid(list(self.node.input), list(self.node.output))]
