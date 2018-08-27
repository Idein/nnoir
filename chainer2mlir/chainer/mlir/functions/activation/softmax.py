from chainer.functions import Softmax
from chainer.mlir.patch import patched_function_apply, patched_function_call

if hasattr(Softmax, 'apply'):
    Softmax.apply = patched_function_apply(Softmax.apply)
else:
    Softmax.__call__ = patched_function_call(Softmax.__call__)

def to_mlir_node(self):
    return {
        b'name': 'Softmax',
        b'params': {
            b'axis': self.axis
        }
    }
Softmax.to_mlir_node = to_mlir_node
