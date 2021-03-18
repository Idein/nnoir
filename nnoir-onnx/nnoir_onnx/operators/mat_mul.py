import numpy as np

from nnoir.functions import MatMul, Linear, Constant
from .utils import Op, gen_unregisterd_node_name, register_node


def gen_value(env, arr):
    name = gen_unregisterd_node_name(env)
    register_node(env, name, arr)

    return name


class OpMatMul(Op):

    def __init__(self, node, *args):
        super(OpMatMul, self).__init__(node, *args)

    def to_function(self, env, constants):
        [x, W] = self.node.input
        if W in constants and len(constants[W].shape) == 2 and np.prod(env[x].shape[1:]) == constants[W].shape[1]:
            return [
                Linear(
                    [x],
                    list(self.node.output),
                    W=env[W].T,
                    b=None
                )
            ]
        elif W in constants:
            const_name = gen_value(env, constants[W])
            nodes = [
                Constant([], [const_name], value=constants[W]),
                MatMul([x, const_name], list(self.node.output))
            ]
            return nodes
        else:
            return [MatMul(list(self.node.input), list(self.node.output))]
