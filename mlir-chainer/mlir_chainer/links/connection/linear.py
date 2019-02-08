from chainer.links import Linear
from mlir_chainer.patch import encode_ndarray, patched_link_call
import mlir.functions as MLIR

Linear.__call__ = patched_link_call(Linear.__call__)

def to_mlir_node(self, inputs, outputs):
    b = encode_ndarray(self.b.data) if (hasattr(self, "b") and self.b is not None) else None
    return MLIR.Linear(
        inputs,
        outputs,
        W=encode_ndarray(self.W.data),
        b=b,
    )
Linear.to_mlir_node = to_mlir_node
