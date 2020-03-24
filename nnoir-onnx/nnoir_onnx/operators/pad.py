import numpy as np
from nnoir.functions import *
from .utils import *


class OpPad(Op):

    def __init__(self, node, *args):
        super(OpPad, self).__init__(node, *args)

        self.mode = b'constant'
        self.pads = None
        self.value = 0.0

    def to_function(self, env, constants):
        if self.opset_version >= 11:
            # pads
            if not self.node.input[1] in constants:
                raise UnsupportedONNXOperation(self.node, 'pads must be "constant"')
            self.pads = constants[self.node.input[1]]

            # optional: constant_value
            if len(self.node.input) >= 3:
                if not self.node.input[2] in constants:
                    raise UnsupportedONNXOperation(self.node, 'constant_value must be "constant"')

                self.value = constants[self.node.input[2]]  # constant_value

            for attr in self.node.attribute:
                if attr.name == 'mode':
                    self.mode = attr.s
                else:
                    raise UnsupportedONNXOperation(self.node, 'unknown attribute {}'.format(attr.s))

            input_ = [self.node.input[0]]
            pads = list(map(int, self.pads))  # In ONNX specification, the type of `pads` is int64
        else:
            # opset version >= 2
            for attr in self.node.attribute:
                if attr.name == 'mode':
                    self.mode = attr.s
                if attr.name == 'pads':
                    self.pads = attr.ints
                if attr.name == 'value':
                    self.value = attr.f

            input_ = list(self.node.input)
            pads = self.pads
        if self.mode != b'constant':
            raise UnsupportedONNXOperation(self.node, 'mode must be "constant"')

        n = len(self.pads) // 2
        return [
            ConstantPadding(
                input_,
                list(self.node.output),
                pads=list(zip(pads[:n], pads[n:])),
                value=self.value,
            )
        ]
