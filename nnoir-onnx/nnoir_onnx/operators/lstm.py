from .utils import *
from nnoir.functions import *

import numpy as np


class OpLSTM(Op):

    def __init__(self, node, *args):
        super(OpLSTM, self).__init__(node, *args)

        self.activation_alpha = []
        self.activation_beta = []
        self.activations = []
        self.clip = 1.0
        self.direction = "forward"
        self.hidden_size = 0
        self.input_forget = 0

        for attr in node.attribute:
            if attr.name == 'activation_alpha':
                self.activation_alpha = attr.floats
            if attr.name == 'activation_beta':
                self.activation_beta = attr.floats
            if attr.name == 'activations':
                self.activations = attr.strs
            if attr.name == 'clip':
                self.clip = attr.f
            if attr.name == 'direction':
                self.direction = attr.s
            if attr.name == 'hidden_size':
                self.hidden_size = attr.i
            if attr.name == 'input_forget':
                self.input_forget = attr.i

    def to_function(self, env, constants):
        num_of_gates = 4
        num_of_peepholes = 3
        num_directions = (env[self.node.input[1]].shape)[0]
        seq_length, batch_size, input_size = env[self.node.input[0]].shape
        hidden_size = self.hidden_size

        if num_directions == 1 and self.direction == "forward":
            graph = []
            init_h = np.zeros((num_directions, batch_size, hidden_size)).astype(np.float32)
            init_c = np.zeros((num_directions, batch_size, hidden_size)).astype(np.float32)
            ps = np.zeros((num_directions, num_of_peepholes*hidden_size)).astype(np.float32)
            li = len(self.node.input)
            if li == 3:
                [x, W, R] = self.node.input
                Ws = np.split(env[W][0], num_of_gates)
                Rs = np.split(env[R][0], num_of_gates)
                WBs = np.zeros((num_of_gates, hidden_size))
                RBs = np.zeros((num_of_gates, hidden_size))
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
                h0 = gen_new_node(env, env[H0])
                c0 = gen_new_node(env, init_c)
                graph += [
                    Constant([], [h0], value=env[H0]),
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
                h0 = gen_new_node(env, env[H0])
                c0 = gen_new_node(env, env[C0])
                graph += [
                    Constant([], [h0], value=env[H0]),
                    Constant([], [c0], value=env[C0]),
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
                h0 = gen_new_node(env, env[H0])
                c0 = gen_new_node(env, env[C0])
                graph += [
                    Constant([], [h0], value=env[H0]),
                    Constant([], [c0], value=env[C0]),
                ]
                Ps = np.split(env[P][0], num_of_peepholes)
            else:
                raise UnsupportedONNXOperation(self.node, 'too many inputs')

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
                raise UnsupportedONNXOperation(self.node, 'too many outputs')

            if seq_length > 1:
                raise UnsupportedONNXOperation(self.node, 'not support LSTM with seq_length > 1')

            # i = sigmoid(np.dot(x, W_i) + np.dot(h0, R_i) + WB_i + RB_i)
            # o = sigmoid(np.dot(x, W_o) + np.dot(h0, R_o) + WB_o + RB_o)
            # f = sigmoid(np.dot(x, W_f) + np.dot(h0, R_f) + WB_f + RB_f)
            # g = tanh(np.dot(x, W_c) + np.dot(h0, R_c) + WB_c + RB_c)
            # c1 = f*c0 + g*i
            # h1 = o*tanh(c1)

            dummy_res = np.zeros((batch_size, hidden_size)).astype(np.float32)

            def gemm(env, x, W, WB, res):
                w = gen_new_node(env, W)
                wb = gen_new_node(env, WB)
                t0 = gen_new_node(env, dummy_res)
                graph = [
                    Constant([], [w], value=W),
                    Constant([], [wb], value=WB),
                    MatMul([x, w], [t0]),
                    Add([t0, wb], [res])
                ]
                return graph

            def gate(env, x, h, W, R, WB, RB, f, res):
                t0 = gen_new_node(env, dummy_res)
                t1 = gen_new_node(env, dummy_res)
                t2 = gen_new_node(env, dummy_res)

                graph = []
                graph += gemm(env, x, W, WB, t0)
                graph += gemm(env, h, R, RB, t1)
                graph += [
                    Add([t0, t1], [t2]),
                    f([t2], [res]),
                ]

                return graph

            i = gen_new_node(env, dummy_res)
            o = gen_new_node(env, dummy_res)
            f = gen_new_node(env, dummy_res)
            g = gen_new_node(env, dummy_res)

            t0 = gen_new_node(env, dummy_res)
            t1 = gen_new_node(env, dummy_res)
            t2 = gen_new_node(env, dummy_res)

            graph += gate(env, x, h0, Ws[0].transpose(), Rs[0].transpose(), WBs[0], RBs[0], Sigmoid, i)
            graph += gate(env, x, h0, Ws[1].transpose(), Rs[1].transpose(), WBs[1], RBs[1], Sigmoid, o)
            graph += gate(env, x, h0, Ws[2].transpose(), Rs[2].transpose(), WBs[2], RBs[2], Sigmoid, f)
            graph += gate(env, x, h0, Ws[3].transpose(), Rs[3].transpose(), WBs[3], RBs[3], Tanh, g)
            graph += [
                Mul([f, c0], [t0]),
                Mul([g, i], [t1]),
                Add([t0, t1], [y_c]),
                Tanh([y_c], [t2]),
                Mul([o, t2], [y_h]),
                Reshape([y_h], [y], shape=(seq_length, num_directions, batch_size, hidden_size))
            ]

            return graph
        else:
            raise UnsupportedONNXOperation(self.node, 'direction is not forward')


def gen_new_node(env, value):
    n = gen_unregisterd_node_name(env)
    register_node(env, n, value)
    return n
