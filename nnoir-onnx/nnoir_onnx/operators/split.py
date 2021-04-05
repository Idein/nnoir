from typing import Tuple

import numpy as np
from nnoir.functions import Constant, MatMul, Transpose

from .utils import Op, gen_unregisterd_node_name, register_node


def create_half_split_matrices(k: int) -> Tuple[np.ndarray, np.ndarray]:
    k_2 = k // 2

    eye = np.eye(k_2, dtype="float32")
    zero = np.zeros((k_2, k_2), dtype="float32")

    return (np.concatenate([eye, zero]), np.concatenate([zero, eye]))


def gen_value(env, arr):
    name = gen_unregisterd_node_name(env)
    register_node(env, name, arr)

    return name


def gen_dummy_value(env, shape):
    dummy_arr = np.zeros(shape).astype(np.float32)
    name = gen_unregisterd_node_name(env)
    register_node(env, name, dummy_arr)

    return name


class OpSplit(Op):
    def __init__(self, node, *args):
        super().__init__(node, *args)

    def to_function(self, env, constants):
        if len(self.node.input) > 1:
            raise Exception("TODO")

        split_axis = 0
        for attr in self.node.attribute:
            if attr.name == "axis":
                split_axis = attr.i

        shape = env[self.node.input[0]].shape
        k = shape[split_axis]
        if k % 2 != 0:
            raise Exception("Cannot reshape on odd size, shape {}, axis {}".format(shape, split_axis))

        matrice_up, matrice_down = create_half_split_matrices(k)
        transpose_perm_0 = list(range(len(shape)))
        transpose_perm_0.append(transpose_perm_0.pop(split_axis))
        transpose_perm_1 = list(range(len(shape)))
        transpose_perm_1.insert(split_axis, transpose_perm_1.pop(-1))

        def transpose_shape(shape, axes):
            l = []
            for axis in axes:
                l.append(shape[axis])
            return tuple(l)

        def linear_shape(x_shape, w_shape):
            """
            (..., n, k), (k, m) -> (batch, channel, n, m)
            """

            n = x_shape[-2]
            k0 = x_shape[-1]
            k1 = w_shape[0]
            m = w_shape[1]
            assert k0 == k1
            return np.concatenate([x_shape[:-2], [n], [m]])

        trans_shape = transpose_shape(shape, transpose_perm_0)
        trans_out = gen_dummy_value(env, trans_shape)

        linear_up_shape = linear_shape(trans_shape, matrice_up.shape)
        linear_up_out = gen_dummy_value(env, linear_up_shape)

        linear_down_shape = linear_shape(trans_shape, matrice_down.shape)
        linear_down_out = gen_dummy_value(env, linear_down_shape)

        up_const = gen_value(env, matrice_up)
        up_const_node = Constant([], [up_const], value=matrice_up)
        down_const = gen_value(env, matrice_up)
        down_const_node = Constant([], [down_const], value=matrice_down)

        transpose_node = Transpose(list(self.node.input), [trans_out], axes=transpose_perm_0)
        linear_up_node = MatMul([trans_out, up_const], [linear_up_out])
        linear_down_node = MatMul([trans_out, down_const], [linear_down_out])
        transpose_up_node = Transpose([linear_up_out], [self.node.output[0]], axes=transpose_perm_1)
        transpose_down_node = Transpose([linear_down_out], [self.node.output[1]], axes=transpose_perm_1)
        nodes = [
            up_const_node,
            down_const_node,
            transpose_node,
            linear_up_node,
            linear_down_node,
            transpose_up_node,
            transpose_down_node,
        ]

        return nodes
