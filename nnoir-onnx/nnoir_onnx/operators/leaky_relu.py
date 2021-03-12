from nnoir.functions import *
from .utils import *


class OpLeakyRelu(Op):

    def __init__(self, node, *args):
        super(OpLeakyRelu, self).__init__(node, *args)

        self.alpha = 0.01
        for attr in node.attribute:
            if attr.name == 'alpha':
                self.alpha = attr.f

    def to_function(self, env, constants):
        return [LeakyReLU(list(self.node.input), list(self.node.output), slope=self.alpha)]
