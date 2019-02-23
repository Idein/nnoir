from chainer.links import Scale
from nnoir_chainer.patch import encode_ndarray, patched_link_call
import nnoir.functions as NNOIR

Scale.__call__ = patched_link_call(Scale.__call__)


def to_nnoir_node(self, inputs, outputs):
    bias_axis = self.bias.axis if (hasattr(self, 'bias') and self.bias is not None) else None
    bias_b = encode_ndarray(self.bias.b.data) if (hasattr(self, 'bias') and self.bias is not None) else None
    return NNOIR.Scale(
        [x.name for x in inputs],
        [x.name for x in outputs],
        axis=self.axis,
        W=encode_ndarray(self.W.data),
        b=bias_b,
    )


Scale.to_nnoir_node = to_nnoir_node
