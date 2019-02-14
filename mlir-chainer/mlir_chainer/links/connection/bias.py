from chainer.links import Bias
from mlir_chainer.patch import encode_ndarray, patched_link_call
import mlir.functions as MLIR

Bias.__call__ = patched_link_call(Bias.__call__)


def to_mlir_node(self, inputs, outputs):
    return MLIR.Bias(inputs, outputs, axis=self.axis, b=encode_ndarray(self.b.data))


Bias.to_mlir_node = to_mlir_node
