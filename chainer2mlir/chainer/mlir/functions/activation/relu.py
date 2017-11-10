from chainer.functions import ReLU
from chainer.mlir.patch import patched_function_apply, patched_function_call

if hasattr(ReLU, 'apply'):
    ReLU.apply = patched_function_apply(ReLU.apply)
else:
    ReLU.__call__ = patched_function_call(ReLU.__call__)

def to_mlir_node(self):
    return {
        b'name': 'ReLU',
        b'params': {}
    }
ReLU.to_mlir_node = to_mlir_node
