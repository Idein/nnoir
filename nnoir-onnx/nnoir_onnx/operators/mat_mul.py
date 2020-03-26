from nnoir.functions import *
from .utils import *


class OpMatMul(Op):

    def __init__(self, node, *args):
        super(OpMatMul, self).__init__(node, *args)

    def to_function(self, env, constants):
        [x, W] = self.node.input
        if W in constants:
            return [
                Linear(
                    [x],
                    list(self.node.output),
                    W=env[W],
                    b=None
                )
            ]
        else:
            raise UnsupportedONNXOperation(self.node, 'B must be constant')
