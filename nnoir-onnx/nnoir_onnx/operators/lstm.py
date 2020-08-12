from .utils import *
from nnoir.functions import *

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
        print("LSTM.to_function start")
        print(self.activation_alpha)
        print(self.activation_beta)
        print(self.activations)
        print(self.clip)
        print(self.direction)
        print(self.hidden_size)
        print(self.input_forget)
        print("LSTM.to_function end")

        if self.direction == "forward":
            # [x, W, R, B, seq_lens, h0, c0, P] = self.node.input
            [x, W, R] = self.node.input
            Ws = np.split(env[W], 4, axis=1)
            Rs = np.split(env[R], 4, axis=1)
            # WB, RB = np.split(env[B], 2)
            # WB_i, WB_o, WB_f, WB_c = np.split(WB, 4)
            # RB_i, RB_o, RB_f, RB_c = np.split(RB, 4)

            h1 = self.node.output[1]
            y = self.node.output[0]

            # i = sigmoid(np.dot(x, W_i) + np.dot(h0, R_i) + WB_i + RB_i)
            # o = sigmoid(np.dot(x, W_o) + np.dot(h0, R_o) + WB_o + RB_o)
            # f = sigmoid(np.dot(x, W_f) + np.dot(h0, R_f) + WB_f + RB_f)
            # g = tanh(np.dot(x, W_c) + np.dot(h0, R_c) + WB_c + RB_c)
            # c1 = f*c0 + g*i
            # h1 = o*tanh(c1)

            def gen_new_node(env):
                n = gen_unregisterd_node_name(env)
                register_node(env, n, None)
                return n

            W_i = gen_new_node(env)
            W_o = gen_new_node(env)
            W_c = gen_new_node(env)

            i0 = gen_new_node(env)
            i  = gen_new_node(env)

            o0 = gen_new_node(env)
            o  = gen_new_node(env)

            g0 = gen_new_node(env)
            g  = gen_new_node(env)

            c1 = gen_new_node(env)

            t0 = gen_new_node(env)

            return [
                Constant([],[W_i],value=Ws[0].transpose((0,2,1))),
                MatMul([x, W_i], [i0]),
                Sigmoid([i0], [i]),
                Constant([],[W_o],value=Ws[1].transpose((0,2,1))),
                MatMul([x, W_o], [o0]),
                Sigmoid([o0], [o]),
                Constant([],[W_c],value=Ws[3].transpose((0,2,1))),
                MatMul([x, W_c], [g0]),
                Tanh([g0], [g]),
                Mul([g, i], [c1]),
                Tanh([c1], [t0]),
                Mul([o, t0], [h1]),
                AddConstant([h1], [y], value=0.0)
            ]
        else:
            raise UnsupportedONNXOperation(self.node, 'direction is not forward')
