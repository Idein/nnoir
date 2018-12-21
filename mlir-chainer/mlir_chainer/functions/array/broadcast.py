from chainer.functions.array.broadcast import BroadcastTo
from mlir_chainer.patch import patched_function_apply, patched_function_call
import mlir.edges as MLIR

if hasattr(BroadcastTo, 'apply'):
    BroadcastTo.apply = patched_function_apply(BroadcastTo.apply)
else:
    BroadcastTo.__call__ = patched_function_call(BroadcastTo.__call__)

def to_mlir_node(self, inputs, outputs):
    return MLIR.BroadcastTo(inputs, outputs)
BroadcastTo.to_mlir_node = to_mlir_node
