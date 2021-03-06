from nnoir.functions import *
from .utils import *


class OpDiv(Op):

    def __init__(self, node, *args):
        super(OpDiv, self).__init__(node, *args)

    def to_function(self, env, constants):
        [a, b] = self.node.input

        def scale(v, w):
            internal_node = gen_unregisterd_node_name(env)
            register_node(env, internal_node, env[w])
            return [
                Constant([], [internal_node], value=constants[w]),
                Mul([v, internal_node], list(self.node.output))
            ]

        if a in constants and b not in constants:
            raise UnsupportedONNXOperation(self.node, 'unimplemented yet')
        elif a not in constants and b in constants:
            return scale(a, 1 / b)
        elif a not in constants and b not in constants:
            return [Div(list(self.node.input), list(self.node.output))]
        else:
            raise UnsupportedONNXOperation(self.node, 'bug! (unreachable here)')
