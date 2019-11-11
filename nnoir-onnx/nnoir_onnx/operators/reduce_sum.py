from nnoir.functions import *
from .utils import *


class OpReduceSum(Op):

    def __init__(self, node):
        super(OpReduceSum, self).__init__(node)

        self.axes = None
        self.keepdims = True
        for attr in node.attribute:
            if attr.name == 'axes':
                self.axes = attr.ints
            if attr.name == 'keepdims':
                self.keepdims = attr.i > 0

    def to_function(self, env, constants):
        [x] = self.node.input
        axes = self.axes
        if axes is None:
            axes = tuple(range(len(x.shape)))
        return [Sum(list(self.node.input), list(self.node.output), axes=axes, keepdims=self.keepdims)]
