from chainer.functions import BroadcastTo
from chainer.mlir.patch import patched_function_apply, patched_function_call

if hasattr(BroadcastTo, 'apply'):
    BroadcastTo.apply = patched_function_apply(BroadcastTo.apply)
else:
    BroadcastTo.__call__ = patched_function_call(BroadcastTo.__call__)

def to_mlir_node(self):
    return {
        b'name': 'BroadcastTo',
        b'params': {}
    }
BroadcastTo.to_mlir_node = to_mlir_node
