from chainer.links import Linear
from chainer.mlir.patch import encode_ndarray, patched_link_call

Linear.__call__ = patched_link_call(Linear.__call__)

def to_mlir_node(self):
    b = encode_ndarray(self.b.data) if (hasattr(self, "b") and self.b is not None) else None
    return {
        b'name': 'Linear',
        b'params': {
            b'W': encode_ndarray(self.W.data),
            b'b': b,
        }
    }
Linear.to_mlir_node = to_mlir_node
