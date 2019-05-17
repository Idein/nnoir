from nnoir.functions import *
from .utils import *


class OpResize(Op):

    def __init__(self, node):
        super(OpResize, self).__init__(node)

    def get_dummy_output(self, env):
        [x, scales] = self.node.input
        x = env[x]
        bc = x.shape[:2]
        others = x.shape[2:]
        scales = env[scales]
        return np.zeros(bc + tuple([int(a*b) for a, b in zip(others, scales)]), dtype=env[x].dtype)

    def to_function(self, env, constants):
        [x, _] = self.node.input
        [y] = self.node.output
        return [
            Resize(
                [x],
                list(self.node.output),
                size=tuple(env[y].shape[2:]),
                mode='align_none'
            )
        ]
