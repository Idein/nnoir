from nnoir.functions import *
from .utils import *


class OpClip(Op):

    def __init__(self, node):
        super(OpClip, self).__init__(node)

    def to_function(self, env, constants):
        _min = -3.4028234663852886e+38
        _max = 3.4028234663852886e+38
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
