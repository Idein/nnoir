from chainer.functions.connection.deconvolution_2d import Deconvolution2DFunction
from chainer.mlir.patch import patched_function_apply, patched_function_call

if hasattr(Deconvolution2DFunction, 'apply'):
    Deconvolution2DFunction.apply = patched_function_apply(Deconvolution2DFunction.apply)
else:
    Deconvolution2DFunction.__call__ = patched_function_call(Deconvolution2DFunction.__call__)

def to_mlir_node(self):
    return {
        b'name': 'Deconvolution2DFunction',
        b'params': {}
    }
Deconvolution2DFunction.to_mlir_node = to_mlir_node
