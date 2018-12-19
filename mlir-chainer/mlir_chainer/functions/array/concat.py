from chainer.functions.array.concat import Concat
from mlir_chainer.patch import patched_function_apply, patched_function_call

if hasattr(Concat, 'apply'):
    Concat.apply = patched_function_apply(Concat.apply)
else:
    Concat.__call__ = patched_function_call(Concat.__call__)

def to_mlir_node(self):
    return {
        b'name': 'Concat',
        b'params': {
            b'axis': self.axis
        }
    }
Concat.to_mlir_node = to_mlir_node
