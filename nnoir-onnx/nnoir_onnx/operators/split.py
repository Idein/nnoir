from typing import List

import numpy as np
from nnoir.functions import Constant, MatMul, Transpose

from .utils import Op, UnsupportedONNXOperation, gen_unregisterd_node_name, register_node


def create_split_matrices(k: int, sizes: List[int]) -> List[np.ndarray]:

    split_matrices = []
    acc = 0
    for sub_k in sizes:
        zero_before = np.zeros((acc, sub_k), dtype="float32")
        eye = np.eye(sub_k, dtype="float32")
        zero_after = np.zeros((k - sub_k - acc, sub_k), dtype="float32")

        split_matrices.append(np.concatenate([zero_before, eye, zero_after]))
        acc += sub_k

    return split_matrices


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
        split_axis = 0
        for attr in self.node.attribute:
            if attr.name == "axis":
                split_axis = attr.i

        output_sizes_on_axis = []
        for output in self.node.output:
            output_sizes_on_axis.append(env[output].shape[split_axis])

        shape = env[self.node.input[0]].shape
        k = shape[split_axis]
        assert k == sum(
            output_sizes_on_axis
        ), f"outputs are not a split of input on for axis {split_axis} of input shape {shape}: {output_sizes_on_axis}"

        matrices = create_split_matrices(k, output_sizes_on_axis)

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

        nodes = [Transpose(list(self.node.input), [trans_out], axes=transpose_perm_0)]
        for i, mat in enumerate(matrices):
            _linear_shape = linear_shape(trans_shape, mat.shape)
            linear_out = gen_dummy_value(env, _linear_shape)

            _const = gen_value(env, mat)
            _const_node = Constant([], [_const], value=mat)
            linear_node = MatMul([trans_out, _const], [linear_out])
            transpose_node = Transpose([linear_out], [self.node.output[i]], axes=transpose_perm_1)
            nodes.extend([_const_node, linear_node, transpose_node])

        return nodes
