from chainer.functions.activation.clipped_relu import ClippedReLU
from mlir_chainer.patch import patched_function_apply, patched_function_call

if hasattr(ClippedReLU, 'apply'):
    ClippedReLU.apply = patched_function_apply(ClippedReLU.apply)
else:
    ClippedReLU.__call__ = patched_function_call(ClippedReLU.__call__)

def to_mlir_node(self):
    return {
        b'name': 'ClippedReLU',
        b'params': {
            b'upper': self.cap
        }
    }
ClippedReLU.to_mlir_node = to_mlir_node
