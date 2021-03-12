from nnoir.functions import *
from .utils import *


class OpPRelu(Op):

    def __init__(self, node, *args):
        super(OpPRelu, self).__init__(node, *args)

    def to_function(self, env, constants):
        [x, slope] = self.node.input

        if slope in constants.keys():
            c = constants[slope].ravel()
            if len(c) != 1:
                raise UnsupportedONNXOperation(self.node, '# of slope size must be 1')
            v = [LeakyReLU([x], list(self.node.output), slope=float(c[0]))]
            return v
        else:
            raise UnsupportedONNXOperation(self.node, '# of slope must be constant')
