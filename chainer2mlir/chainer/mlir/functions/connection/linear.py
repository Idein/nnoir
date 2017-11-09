from chainer.functions.connection.linear import LinearFunction
from chainer.mlir.patch import patched_function_apply, patched_function_call

if hasattr(LinearFunction, 'apply'):
    LinearFunction.apply = patched_function_apply(LinearFunction.apply)
else:
    LinearFunction.__call__ = patched_function_call(LinearFunction.__call__)

def to_mlir_node(self):
    return {
        b'name': 'LinearFunction',
        b'params': {}
    }
LinearFunction.to_mlir_node = to_mlir_node
