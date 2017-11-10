from chainer.functions import Reshape
from chainer.mlir.patch import patched_function_apply, patched_function_call

if hasattr(Reshape, 'apply'):
    Reshape.apply = patched_function_apply(Reshape.apply)
else:
    Reshape.__call__ = patched_function_call(Reshape.__call__)

def to_mlir_node(self):
    return {
        b'name': 'Reshape',
        b'params': {
            b'shape': self.shape
        }
    }
Reshape.to_mlir_node = to_mlir_node
