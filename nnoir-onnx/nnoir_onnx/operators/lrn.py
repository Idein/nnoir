from nnoir.functions import *
from .utils import *


class OpLRN(Op):

    def __init__(self, node, *args):
        super(OpLRN, self).__init__(node, *args)

        self.alpha = 0.0001
        self.beta = 0.75
        self.bias = 1.0
        self.size = None
        for attr in node.attribute:
            if attr.name == 'alpha':
                self.alpha = attr.f
            if attr.name == 'beta':
                self.beta = attr.f
            if attr.name == 'bias':
                self.bias = attr.f
            if attr.name == 'size':
                self.size = attr.i

    def to_function(self, env, constants):
        return [
            LocalResponseNormalization(
                list(self.node.input),
                list(self.node.output),
                n=self.size,
                k=self.bias,
                alpha=self.alpha / self.size,
                beta=self.beta
            )
        ]
