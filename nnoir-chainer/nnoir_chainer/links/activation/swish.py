from chainer.links.activation.swish import Swish
from nnoir_chainer.patch import encode_ndarray, patched_link_call
import nnoir.functions as NNOIR

Swish.__call__ = patched_link_call(Swish.__call__)


def to_nnoir_node(self, inputs, outputs):
    return NNOIR.Swish([x.name for x in inputs], [x.name for x in outputs], beta=encode_ndarray(self.beta.data))


Swish.to_nnoir_node = to_nnoir_node
