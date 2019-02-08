from chainer.functions.connection.convolution_2d import Convolution2DFunction
from mlir_chainer.patch import patched_function_apply, patched_function_call
import mlir.functions as MLIR

if hasattr(Convolution2DFunction, 'apply'):
    Convolution2DFunction.apply = patched_function_apply(Convolution2DFunction.apply)
else:
    Convolution2DFunction.__call__ = patched_function_call(Convolution2DFunction.__call__)

def to_mlir_node(self, inputs, outputs):
    return MLIR.Convolution2DFunction(inputs, outputs)
Convolution2DFunction.to_mlir_node = to_mlir_node
