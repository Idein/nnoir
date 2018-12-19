from chainer.functions.activation.leaky_relu import LeakyReLU
from chainer.mlir.patch import patched_function_apply, patched_function_call

if hasattr(LeakyReLU, 'apply'):
    LeakyReLU.apply = patched_function_apply(LeakyReLU.apply)
else:
    LeakyReLU.__call__ = patched_function_call(LeakyReLU.__call__)

def to_mlir_node(self):
    return {
        b'name': 'LeakyReLU',
        b'params': {
            b'slope': self.slope
        }
    }
LeakyReLU.to_mlir_node = to_mlir_node
