from chainer.links import Linear
from nnoir_chainer.patch import encode_ndarray, patched_link_call
import nnoir.functions as NNOIR

Linear.__call__ = patched_link_call(Linear.__call__)


def to_nnoir_node(self, inputs, outputs):
    b = encode_ndarray(self.b.data) if (hasattr(self, "b") and self.b is not None) else None
    return NNOIR.Linear(
        [x.name for x in inputs],
        [x.name for x in outputs],
        W=encode_ndarray(self.W.data),
        b=b,
    )


Linear.to_nnoir_node = to_nnoir_node
