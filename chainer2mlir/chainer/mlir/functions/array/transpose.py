from chainer.functions import Transpose
from chainer.mlir.patch import patched_function_apply, patched_function_call

if hasattr(Transpose, 'apply'):
    Transpose.apply = patched_function_apply(Transpose.apply)
else:
    Transpose.__call__ = patched_function_call(Transpose.__call__)

def to_mlir_node(self):
    return {
        b'name': 'Transpose',
        b'params': {
            b'axes': self.shape
        }
    }
Transpose.to_mlir_node = to_mlir_node
