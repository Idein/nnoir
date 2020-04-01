from nnoir.functions import *
from .utils import *


class OpAdd(Op):

    def __init__(self, node, *args):
        super(OpAdd, self).__init__(node, *args)

    def to_function(self, env, constants):
        [a, b] = self.node.input

        def constant_add(v, w):
            internal_node = gen_unregisterd_node_name(env)
            register_node(env, internal_node, env[w])
            return [
                Constant([], [internal_node], value=constants[w]),
                Add([v, internal_node], list(self.node.output))
            ]

        if a in constants and b not in constants:
            return constant_add(b, a)
        elif a not in constants and b in constants:
            return constant_add(a, b)
        elif a not in constants and b not in constants:
            return [Add(list(self.node.input), list(self.node.output))]
        else:
            raise UnsupportedONNXOperation(self.node, 'bug! (unreachable here)')
