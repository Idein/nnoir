from nnoir.functions import *
from .utils import *


class OpMatMul(Op):

    def __init__(self, node):
        super(OpMatMul, self).__init__(node)

    def get_dummy_output(self, env):
        [x, W] = self.node.input
        return env[x].dot(env[W])

    def to_function(self, env, constants):
        [x, W] = self.node.input
        if W in constants:
            return [
                Linear(
                    [x],
                    list(self.node.output),
                    W=encode_ndarray(env[W]),
                    b=None
                )
            ]
        else:
            raise UnsupportedONNXOperation(self.node, 'W must be constant')
