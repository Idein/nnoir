from typing import Tuple
import numpy as np

from nnoir.functions import Transpose, Linear
from .utils import Op, gen_unregisterd_node_name, register_node


def create_half_split_matrices(k: int) -> Tuple[np.ndarray, np.ndarray]:
    k_2 = k // 2

    eye = np.eye(k_2, dtype="float32")
    zero = np.zeros((k_2, k_2), dtype="float32")

    return (np.concatenate([eye, zero]), np.concatenate([zero, eye]))


def gen_dummy_value(env, shape):
    dummy_arr = np.zeros(shape)
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

        shape = self.onnx.nodes[self.node.input[0]].shape
        k = shape[split_axis]
        if k % 2 != 0:
            raise Exception("Cannot reshape on odd size, shape {}, axis {}".format(shape, split_axis))

        matrice_up, matrice_down = create_half_split_matrices(k)
        matrice_up, matrice_down = matrice_up.T, matrice_down.T
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
            (batch, channel, n, k), (m, k) -> (batch, channel, n, m)
            """

            assert x_shape[3] == w_shape[1]
            return ([x_shape[0], x_shape[1], x_shape[2], w_shape[0]])

        trans_shape = transpose_shape(shape, transpose_perm_0)
        trans_out = gen_dummy_value(env, trans_shape)

        linear_up_shape = linear_shape(trans_shape, matrice_up.shape)
        linear_up_out = gen_dummy_value(env, linear_up_shape)

        linear_down_shape = linear_shape(trans_shape, matrice_down.shape)
        linear_down_out = gen_dummy_value(env, linear_down_shape)

        transpose_node = Transpose(list(self.node.input), [trans_out], axes=transpose_perm_0)
        linear_up_node = Linear([trans_out], [linear_up_out], W=matrice_up, b=None)
        linear_down_node = Linear([trans_out], [linear_down_out], W=matrice_down, b=None)
        transpose_up_node = Transpose([linear_up_out], [self.node.output[0]], axes=transpose_perm_1)
        transpose_down_node = Transpose([linear_down_out], [self.node.output[1]], axes=transpose_perm_1)
        nodes = [
            transpose_node,
            linear_up_node,
            linear_down_node,
            transpose_up_node,
            transpose_down_node
        ]

        return nodes
