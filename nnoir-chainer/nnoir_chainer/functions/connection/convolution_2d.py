from chainer.functions.connection.convolution_2d import Convolution2DFunction
from nnoir_chainer.patch import patched_function_apply, patched_function_call
import nnoir.functions as NNOIR

if hasattr(Convolution2DFunction, 'apply'):
    Convolution2DFunction.apply = patched_function_apply(Convolution2DFunction.apply)
else:
    Convolution2DFunction.__call__ = patched_function_call(Convolution2DFunction.__call__)


def to_nnoir_node(self, inputs, outputs):
    return NNOIR.Convolution2DFunction(inputs, outputs)


Convolution2DFunction.to_nnoir_node = to_nnoir_node
