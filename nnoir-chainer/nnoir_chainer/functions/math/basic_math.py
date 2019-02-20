from chainer.functions.math.basic_math import Add, AddConstant, Mul, MulConstant
from nnoir_chainer.patch import patched_function_apply, patched_function_call
import nnoir.functions as NNOIR

if hasattr(Add, 'apply'):
    Add.apply = patched_function_apply(Add.apply)
else:
    Add.__call__ = patched_function_call(Add.__call__)


def to_nnoir_node(self, inputs, outputs):
    return NNOIR.Add(inputs, outputs)


Add.to_nnoir_node = to_nnoir_node

if hasattr(AddConstant, 'apply'):
    AddConstant.apply = patched_function_apply(AddConstant.apply)
else:
    AddConstant.__call__ = patched_function_call(AddConstant.__call__)


def to_nnoir_node(self, inputs, outputs):
    return NNOIR.AddConstant(inputs, outputs, value=float(self.value))


AddConstant.to_nnoir_node = to_nnoir_node

if hasattr(Mul, 'apply'):
    Mul.apply = patched_function_apply(Mul.apply)
else:
    Mul.__call__ = patched_function_call(Mul.__call__)


def to_nnoir_node(self, inputs, outputs):
    return NNOIR.Mul(inputs, outputs)


Mul.to_nnoir_node = to_nnoir_node

if hasattr(MulConstant, 'apply'):
    MulConstant.apply = patched_function_apply(MulConstant.apply)
else:
    MulConstant.__call__ = patched_function_call(MulConstant.__call__)


def to_nnoir_node(self, inputs, outputs):
    return NNOIR.MulConstant(inputs, outputs, value=float(self.value))


MulConstant.to_nnoir_node = to_nnoir_node
