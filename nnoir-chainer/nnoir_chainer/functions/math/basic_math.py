from chainer.functions.math.basic_math import Add, AddConstant, Sub, Mul, MulConstant
from nnoir_chainer.patch import patched_function_apply, patched_function_call
import nnoir.functions as NNOIR

if hasattr(Add, 'apply'):
    Add.apply = patched_function_apply(Add.apply)
else:
    Add.__call__ = patched_function_call(Add.__call__)


def to_nnoir_node(self, inputs, outputs):
    return NNOIR.Add([x.name for x in inputs], [x.name for x in outputs])


Add.to_nnoir_node = to_nnoir_node

if hasattr(AddConstant, 'apply'):
    AddConstant.apply = patched_function_apply(AddConstant.apply)
else:
    AddConstant.__call__ = patched_function_call(AddConstant.__call__)


def to_nnoir_node(self, inputs, outputs):
    return NNOIR.AddConstant([x.name for x in inputs], [x.name for x in outputs], value=float(self.value))


AddConstant.to_nnoir_node = to_nnoir_node

if hasattr(Sub, 'apply'):
    Sub.apply = patched_function_apply(Sub.apply)
else:
    Sub.__call__ = patched_function_call(Sub.__call__)


def to_nnoir_node(self, inputs, outputs):
    return NNOIR.Sub([x.name for x in inputs], [x.name for x in outputs])


Sub.to_nnoir_node = to_nnoir_node

if hasattr(Mul, 'apply'):
    Mul.apply = patched_function_apply(Mul.apply)
else:
    Mul.__call__ = patched_function_call(Mul.__call__)


def to_nnoir_node(self, inputs, outputs):
    return NNOIR.Mul([x.name for x in inputs], [x.name for x in outputs])


Mul.to_nnoir_node = to_nnoir_node

if hasattr(MulConstant, 'apply'):
    MulConstant.apply = patched_function_apply(MulConstant.apply)
else:
    MulConstant.__call__ = patched_function_call(MulConstant.__call__)


def to_nnoir_node(self, inputs, outputs):
    return NNOIR.MulConstant([x.name for x in inputs], [x.name for x in outputs], value=float(self.value))


MulConstant.to_nnoir_node = to_nnoir_node
