import numpy as np
from nnoir.functions import *
from .utils import *


class OpPad(Op):

    def __init__(self, node):
        super(OpPad, self).__init__(node)

        self.mode = b'constant'
        self.pads = None
        self.value = 0.0
        for attr in node.attribute:
            if attr.name == 'mode':
                self.mode = attr.s
            if attr.name == 'pads':
                self.pads = attr.ints
            if attr.name == 'value':
                self.value = attr.f

        if self.mode != b'constant':
            raise UnsupportedONNXOperation(self.node, 'mode must be "constant"')

    def to_function(self, env, constants):
        n = len(self.pads) // 2
        return [
            ConstantPadding(
                list(self.node.input),
                list(self.node.output),
                pads=list(zip(self.pads[:n], self.pads[n:])),
                value=self.value,
            )
        ]
