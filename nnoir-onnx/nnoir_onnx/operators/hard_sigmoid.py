from nnoir.functions import *

from .utils import *


class OpHardSwish(Op):
    def __init__(self, node, *args):
        super(OpHardSwish, self).__init__(node, *args)

    def to_function(self, env, constants):
        [x] = self.node.input
        t0 = gen_unregisterd_node_name(env)
        register_node(env, t0, env[x])
        t1 = gen_unregisterd_node_name(env)
        register_node(env, t1, env[x])
        t2 = gen_unregisterd_node_name(env)
        register_node(env, t2, env[x])

        return [
            AddConstant([x], [t0], value=3.0),
            ClippedReLU([t0], [t1], upper=6.0),
            Mul([x, t1], [t2]),
            MulConstant([t2], list(self.node.output), value=1 / 6),
        ]
