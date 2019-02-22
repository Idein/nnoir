from chainer.functions.connection.linear import LinearFunction
from nnoir_chainer.patch import patched_function_apply, patched_function_call

if hasattr(LinearFunction, 'apply'):
    LinearFunction.apply = patched_function_apply(LinearFunction.apply)
else:
    LinearFunction.__call__ = patched_function_call(LinearFunction.__call__)
