from chainer.functions.connection.convolution_2d import Convolution2DFunction
from nnoir_chainer.patch import patched_function_apply, patched_function_call

if hasattr(Convolution2DFunction, 'apply'):
    Convolution2DFunction.apply = patched_function_apply(Convolution2DFunction.apply)
else:
    Convolution2DFunction.__call__ = patched_function_call(Convolution2DFunction.__call__)
