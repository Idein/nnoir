from nnoir.functions import *

from .utils import *


class OpHardSwish(Op):
    def __init__(self, node, *args):
        super(OpHardSwish, self).__init__(node, *args)

    def to_function(self, env, constants):
        [x] = self.node.input
        t0 = gen_unregisterd_node_name(env)
        register_node(env, t0, np.array(3.0, dtype=np.float32))
        t1 = gen_unregisterd_node_name(env)
        register_node(env, t1, env[x])
        t2 = gen_unregisterd_node_name(env)
        register_node(env, t2, env[x])
        t3 = gen_unregisterd_node_name(env)
        register_node(env, t3, env[x])
        t4 = gen_unregisterd_node_name(env)
        register_node(env, t4, np.array(6.0, dtype=np.float32))

        return [
            Constant([], [t0], value=np.array(3.0, dtype=np.float32)),
            Add([x, t0], [t1]),
            ClippedReLU([t1], [t2], upper=6.0),
            Mul([x, t2], [t3]),
            Constant([], [t4], value=np.array(6.0, dtype=np.float32)),
            Div([t3, t4], list(self.node.output)),
        ]
