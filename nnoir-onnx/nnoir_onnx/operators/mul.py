import numpy as np
from nnoir.functions import *
from .utils import *


class OpMul(Op):

    def __init__(self, node, *args):
        super().__init__(node, *args)

    def to_function(self, env, constants):
        [a, b] = self.node.input
        [y] = self.node.output

        def scale(v, w):
            # use unidirectional broadcasting rule
            axis = env[v].ndim - env[w].ndim
            if axis == 0 and not unidirectional_broadcastable(env[v].shape, env[w].shape):

                shape = env[y].shape
                internal_node_val = np.broadcast_to(env[v], shape=shape)
                internal_node = gen_unregisterd_node_name(env)
                register_node(env, internal_node, internal_node_val)
                return [
                    BroadcastTo([v], [internal_node], shape=shape),
                    Scale([internal_node], list(self.node.output), axis=axis, W=encode_ndarray(constants[w]), b=None)
                ]
            else:
                return [Scale([v], list(self.node.output), axis=axis, W=encode_ndarray(constants[w]), b=None)]

        if a in constants and b not in constants:
            return scale(b, a)
        elif a not in constants and b in constants:
            return scale(a, b)
        elif a not in constants and b not in constants:
            return [Mul(list(self.node.input), list(self.node.output))]
        else:
            raise UnsupportedONNXOperation(self.node, 'bug! (unreachable here)')
