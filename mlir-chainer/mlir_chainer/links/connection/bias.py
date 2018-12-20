from chainer.links import Bias
from mlir_chainer.patch import encode_ndarray, patched_link_call

Bias.__call__ = patched_link_call(Bias.__call__)

def to_mlir_node(self):
    return {
        b'name': 'Bias',
        b'params': {
            b'axis': self.axis,
            b'b': encode_ndarray(self.b.data),
        }
    }
Bias.to_mlir_node = to_mlir_node
