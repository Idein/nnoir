from chainer.links import Bias
from nnoir_chainer.patch import encode_ndarray, patched_link_call
import nnoir.functions as NNOIR

Bias.__call__ = patched_link_call(Bias.__call__)


def to_nnoir_node(self, inputs, outputs):
    return NNOIR.Bias([x.name for x in inputs], [x.name for x in outputs], axis=self.axis, b=encode_ndarray(self.b.data))


Bias.to_nnoir_node = to_nnoir_node
