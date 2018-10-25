from chainer.functions.activation.sigmoid import Sigmoid
from chainer.mlir.patch import patched_function_apply, patched_function_call

if hasattr(Sigmoid, 'apply'):
    Sigmoid.apply = patched_function_apply(Sigmoid.apply)
else:
    Sigmoid.__call__ = patched_function_call(Sigmoid.__call__)

def to_mlir_node(self):
    return {
        b'name': 'Sigmoid',
        b'params': {}
    }
Sigmoid.to_mlir_node = to_mlir_node
