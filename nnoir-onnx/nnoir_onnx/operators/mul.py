from nnoir.functions import *
from .utils import *


class OpMul(Op):

    def __init__(self, node, *args):
        super().__init__(node, *args)


        self._axis = None

        for attr in node.attribute:
            # opset version 1
            if attr.name == 'consumed_inputs':
                pass  # legacy attribute

            # opset version <= 6
            elif attr.name == 'broadcast':
                # assume both inputs have the same shape if broadcast is disabled
                pass
            elif attr.name == 'axis':
                self._axis = attr.i

            else:
                raise UnsupportedONNXOperation(self.node, 'unknown attribute {}'.format(attr.name))

    def to_function(self, env, constants):
        [a, b] = self.node.input

        def scale(v, w):
            if self._axis is None:
                # use unidirectional broadcasting rule
                self._axis = env[v].ndim - env[w].ndim
                if self._axis == 0 and not unidirectional_broadcastable(env[v].shape, env[w].shape):
                    raise UnsupportedONNXOperation(
                        self.node,
                        'shape {} is not unidirectionnally broadcastable to shape {}'.format(env[w].shape, env[v].shape))

            return [Scale([v], list(self.node.output), axis=self._axis, W=encode_ndarray(constants[w]), b=None)]

        if a in constants and b not in constants:
            return scale(b, a)
        elif a not in constants and b in constants:
            return scale(a, b)
        elif a not in constants and b not in constants:
            if self._axis is not None:
                a_shape = env[a].shape
                b_shape = env[b].shape
                shape_post_len = len(a_shape) - self._axis - len(b_shape)
                shape = (1,)*self._axis + b_shape + (1,)*shape_post_len

                
                return []
            return [Mul(list(self.node.input), list(self.node.output))]
        else:
            raise UnsupportedONNXOperation(self.node, 'bug! (unreachable here)')
