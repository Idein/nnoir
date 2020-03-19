from nnoir.functions import *
from .utils import *


class OpClip(Op):

    def __init__(self, node, *args):
        super(OpClip, self).__init__(node, *args)

    def to_function(self, env, constants):
        _min = -3.4028234663852886e+38
        _max = 3.4028234663852886e+38

        if self.opset_version < 6:
            raise UnsupportedONNXOperation(self.node, 'only opset_version >= 6 is supported')

        if self.opset_version >= 11:
            if len(self.node.input) >= 2:
                _min = constants[self.node.input[1]]
            if len(self.node.input) >= 3:
                _max = constants[self.node.input[2]]

            if _min != 0.0:
                raise UnsupportedONNXOperation(self.node, 'min must be 0.0')

            return [
                ClippedReLU(
                    [self.node.input[0]],
                    list(self.node.output),
                    upper=_max
                )
            ]

        else:
            # opset_version 6
            for attr in self.node.attribute:
                if attr.name == 'max':
                    _max = attr.f
                if attr.name == 'min':
                    _min = attr.f

            if _min != 0.0:
                raise UnsupportedONNXOperation(self.node, 'min must be 0.0')

            return [
                ClippedReLU(
                    list(self.node.input),
                    list(self.node.output),
                    upper=_max
                )
            ]
