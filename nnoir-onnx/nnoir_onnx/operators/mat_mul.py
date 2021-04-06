import numpy as np
from nnoir.functions import Constant, Linear, MatMul

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
        if W in constants and constants[W].ndim == 2 and env[x].ndim == 2:
            return [Linear([x], list(self.node.output), W=env[W].T, b=None)]
        elif W in constants:
            const_name = gen_value(env, constants[W])
            nodes = [
                Constant([], [const_name], value=constants[W]),
                MatMul([x, const_name], list(self.node.output)),
            ]
            return nodes
        else:
            return [MatMul(list(self.node.input), list(self.node.output))]
