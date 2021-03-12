from nnoir.functions import *
from .utils import *


class OpElu(Op):

    def __init__(self, node, *args):
        super(OpElu, self).__init__(node, *args)

        self.alpha = 1.
        for attr in self.node.attribute:
            if attr.name == 'alpha':
                self.alpha = attr.f

    def to_function(self, env, constants):
        return [
            ELU(
                list(self.node.input),
                list(self.node.output),
                alpha=self.alpha
            )
        ]
