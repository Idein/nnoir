from typing import Any, Dict, List, Optional, Tuple, no_type_check

import numpy as np
import onnx
from nnoir.functions import Add, Concat, Constant, Function, Linear, Mul, ReLU, Reshape, Sigmoid, Sum, Tanh, Transpose
from numpy.typing import NDArray

from .utils import *


class OpLSTM(Op):
    def __init__(self, node: onnx.NodeProto, *args: Any):
        super(OpLSTM, self).__init__(node, *args)

        self.activation_alpha = []
        self.activation_beta = []
        self.activations = []
        self.clip = 1.0
        self.direction = "forward"
        self.hidden_size = 0
        self.input_forget = 0

        for attr in node.attribute:
            if attr.name == "activation_alpha":
                self.activation_alpha = attr.floats
            if attr.name == "activation_beta":
                self.activation_beta = attr.floats
            if attr.name == "activations":
                self.activations = attr.strings
            if attr.name == "clip":
                self.clip = attr.f
            if attr.name == "direction":
                self.direction = attr.s
            if attr.name == "hidden_size":
                self.hidden_size = attr.i
            if attr.name == "input_forget":
                self.input_forget = attr.i

    @no_type_check
    def to_function(self, env: Dict[str, NDArray[Any]], constants: Dict[str, NDArray[Any]]) -> List[Function]:
        num_of_gates = 4
        num_of_peepholes = 3
        num_directions = (env[self.node.input[1]].shape)[0]
        seq_length, batch_size, input_size = env[self.node.input[0]].shape
        hidden_size = self.hidden_size

        if seq_length > 1:
            raise UnsupportedONNXOperation(self.node, "not support LSTM with seq_length > 1")

        if num_directions == 1 and self.direction == "forward":
            l = len(self.activations)
            if l == 0:
                activation_f, activation_g, activation_h = [Sigmoid, Tanh, Tanh]
            elif l == 3:

                def to_op(s: bytes) -> Function:
                    if s == b"Sigmoid":
                        return Sigmoid
                    elif s == b"Tanh":
                        return Tanh
                    elif s == b"Relu":
                        return ReLU
                    else:
                        raise UnsupportedONNXOperation(self.node, f"{s!r} is not supported")

                activation_f, activation_g, activation_h = [to_op(f) for f in self.activations]
            else:
                raise UnsupportedONNXOperation(self.node, "the number of activations must be 0 or 3")

            graph: List[Function] = []
            init_h = np.zeros((batch_size, hidden_size)).astype(np.float32)
            init_c = np.zeros((batch_size, hidden_size)).astype(np.float32)
            ps = np.zeros((num_of_peepholes, hidden_size)).astype(np.float32)
            li = len(self.node.input)
            if li == 3:
                [x, W, R] = self.node.input
                Ws = np.split(env[W][0], num_of_gates)
                Rs = np.split(env[R][0], num_of_gates)
                WBs = np.zeros((num_of_gates, hidden_size)).astype(np.float32)
                RBs = np.zeros((num_of_gates, hidden_size)).astype(np.float32)
                sequence_lens = np.repeat(seq_length, batch_size).astype(np.int32)
                h0 = gen_new_node(env, init_h)
                c0 = gen_new_node(env, init_c)
                graph += [
                    Constant([], [h0], value=init_h),
                    Constant([], [c0], value=init_c),
                ]
                Ps = ps
            elif li == 4:
                [x, W, R, B] = self.node.input
                Ws = np.split(env[W][0], num_of_gates)
                Rs = np.split(env[R][0], num_of_gates)
                WB, RB = np.split(env[B][0], 2)
                WBs = np.split(WB, num_of_gates)
                RBs = np.split(RB, num_of_gates)
                sequence_lens = np.repeat(seq_length, batch_size).astype(np.int32)
                h0 = gen_new_node(env, init_h)
                c0 = gen_new_node(env, init_c)
                graph += [
                    Constant([], [h0], value=init_h),
                    Constant([], [c0], value=init_c),
                ]
                Ps = ps
            elif li == 5:
                [x, W, R, B, seq_lens] = self.node.input
                Ws = np.split(env[W][0], num_of_gates)
                Rs = np.split(env[R][0], num_of_gates)
                WB, RB = np.split(env[B][0], 2)
                WBs = np.split(WB, num_of_gates)
                RBs = np.split(RB, num_of_gates)
                sequence_lens = env[seq_lens]
                h0 = gen_new_node(env, init_h)
                c0 = gen_new_node(env, init_c)
                graph += [
                    Constant([], [h0], value=init_h),
                    Constant([], [c0], value=init_c),
                ]
                Ps = ps
            elif li == 6:
                [x, W, R, B, seq_lens, H0] = self.node.input
                Ws = np.split(env[W][0], num_of_gates)
                Rs = np.split(env[R][0], num_of_gates)
                WB, RB = np.split(env[B][0], 2)
                WBs = np.split(WB, num_of_gates)
                RBs = np.split(RB, num_of_gates)
                sequence_lens = env[seq_lens]
                h0 = gen_new_node(env, env[H0][0])
                c0 = gen_new_node(env, init_c)
                if H0 in constants:
                    graph += [Constant([], [h0], value=env[H0][0])]
                else:
                    graph += [
                        Reshape([H0], [h0], shape=(batch_size, hidden_size)),
                    ]
                graph += [
                    Constant([], [c0], value=init_c),
                ]
                Ps = ps
            elif li == 7:
                [x, W, R, B, seq_lens, H0, C0] = self.node.input
                Ws = np.split(env[W][0], num_of_gates)
                Rs = np.split(env[R][0], num_of_gates)
                WB, RB = np.split(env[B][0], 2)
                WBs = np.split(WB, num_of_gates)
                RBs = np.split(RB, num_of_gates)
                sequence_lens = env[seq_lens]
                h0 = gen_new_node(env, env[H0][0])
                c0 = gen_new_node(env, env[C0][0])
                if H0 in constants:
                    graph += [Constant([], [h0], value=env[H0][0])]
                else:
                    graph += [
                        Reshape([H0], [h0], shape=(batch_size, hidden_size)),
                    ]
                if C0 in constants:
                    graph += [Constant([], [c0], value=env[C0][0])]
                else:
                    graph += [
                        Reshape([C0], [c0], shape=(batch_size, hidden_size)),
                    ]
                Ps = ps
            elif li == 8:
                [x, W, R, B, seq_lens, H0, C0, P] = self.node.input
                Ws = np.split(env[W][0], num_of_gates)
                Rs = np.split(env[R][0], num_of_gates)
                WB, RB = np.split(env[B][0], 2)
                WBs = np.split(WB, num_of_gates)
                RBs = np.split(RB, num_of_gates)
                sequence_lens = env[seq_lens]
                h0 = gen_new_node(env, env[H0][0])
                c0 = gen_new_node(env, env[C0][0])
                if H0 in constants:
                    graph += [Constant([], [h0], value=env[H0][0])]
                else:
                    graph += [
                        Reshape([H0], [h0], shape=(batch_size, hidden_size)),
                    ]
                if C0 in constants:
                    graph += [Constant([], [c0], value=env[C0][0])]
                else:
                    graph += [
                        Reshape([C0], [c0], shape=(batch_size, hidden_size)),
                    ]
                Ps = np.split(env[P][0], num_of_peepholes)
            else:
                raise UnsupportedONNXOperation(self.node, "too many inputs")

            lo = len(self.node.output)
            dummy_cell = np.zeros((num_directions, batch_size, hidden_size)).astype(np.float32)
            if lo == 1:
                [y] = self.node.output  # (seq_length, num_directions, batch_size, hidden_size)
                y_h = gen_new_node(env, dummy_cell)
                y_c = gen_new_node(env, dummy_cell)
            elif lo == 2:
                [y, y_h] = self.node.output
                y_c = gen_new_node(env, dummy_cell)
            elif lo == 3:
                [y, y_h, y_c] = self.node.output
            elif lo == 0:
                return []
            else:
                raise UnsupportedONNXOperation(self.node, "too many outputs")

            # i = sigmoid(np.dot(x, W_i) + np.dot(h0, R_i) + WB_i + RB_i + P_i*c0)
            # f = sigmoid(np.dot(x, W_f) + np.dot(h0, R_f) + WB_f + RB_f + P_f*c0)
            # g = tanh(np.dot(x, W_c) + np.dot(h0, R_c) + WB_c + RB_c)
            # c1 = f*c0 + g*i
            # o = sigmoid(np.dot(x, W_o) + np.dot(h0, R_o) + WB_o + RB_o + P_o*c1)
            # h1 = o*tanh(c1)

            x0 = gen_new_node(env, env[x].reshape((batch_size, input_size)))
            graph += [Reshape([x], [x0], shape=(batch_size, input_size))]

            dummy_res = np.zeros((batch_size, hidden_size)).astype(np.float32)

            def gate(env, res, x, h, W, R, WB, RB, f, c=None, P=None) -> List[Function]:
                t0 = gen_new_node(env, dummy_res)
                t1 = gen_new_node(env, dummy_res)
                t2 = gen_new_node(env, dummy_res)

                graph: List[Function] = []
                graph += [Linear([x], [t0], W=W, b=WB)]
                graph += [Linear([h], [t1], W=R, b=RB)]
                graph += [Add([t0, t1], [t2])]
                if P is not None:
                    t3 = gen_new_node(env, dummy_res)
                    t4 = gen_new_node(env, dummy_res)
                    p = gen_new_node(env, P)
                    graph += [
                        Constant([], [p], value=P),
                        Mul([p, c], [t3]),
                        Add([t2, t3], [t4]),
                        f([t4], [res]),
                    ]
                else:
                    graph += [f([t2], [res])]

                return graph

            i = gen_new_node(env, dummy_res)
            o = gen_new_node(env, dummy_res)
            f = gen_new_node(env, dummy_res)
            g = gen_new_node(env, dummy_res)

            t0 = gen_new_node(env, dummy_res)
            t1 = gen_new_node(env, dummy_res)
            t2 = gen_new_node(env, dummy_res)

            res_c = gen_new_node(env, dummy_res)
            res_h = gen_new_node(env, dummy_res)

            graph += gate(env, i, x0, h0, Ws[0], Rs[0], WBs[0], RBs[0], activation_f, c0, Ps[0])
            graph += gate(env, f, x0, h0, Ws[2], Rs[2], WBs[2], RBs[2], activation_f, c0, Ps[2])
            graph += gate(env, g, x0, h0, Ws[3], Rs[3], WBs[3], RBs[3], activation_g)
            graph += [
                Mul([f, c0], [t0]),
                Mul([g, i], [t1]),
                Add([t0, t1], [res_c]),
                Reshape([res_c], [y_c], shape=dummy_cell.shape),
            ]
            graph += gate(env, o, x0, h0, Ws[1], Rs[1], WBs[1], RBs[1], activation_f, res_c, Ps[1])
            graph += [
                activation_h([res_c], [t2]),
                Mul([o, t2], [res_h]),
                Reshape([res_h], [y_h], shape=dummy_cell.shape),
                Reshape(
                    [res_h],
                    [y],
                    shape=(seq_length, num_directions, batch_size, hidden_size),
                ),
            ]

            return graph
        else:
            raise UnsupportedONNXOperation(self.node, "direction is not forward")


def gen_new_node(env: Dict[str, NDArray[Any]], value: NDArray[Any]) -> str:
    n = gen_unregisterd_node_name(env)
    register_node(env, n, value)
    return n
