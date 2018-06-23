from chainer.functions import Pad
from chainer.mlir.patch import patched_function_apply, patched_function_call
from numpy.lib.arraypad import _validate_lengths # dirty

if hasattr(Pad, 'apply'):
    Pad.apply = patched_function_apply(Pad.apply)
else:
    Pad.__call__ = patched_function_call(Pad.__call__)

def to_mlir_node(self):
    [var] = self.chainer_input_variables
    if self.mode == 'constant':
        return {
            b'name': 'ConstantPadding',
            b'params': {
                b'pads': _validate_lengths(var.data, self.pad_width),
                b'value': float(self.keywords['constant_values'])
            }
        }
    else:
        raise Exception('unsupported pad mode "{}"'.format(self.mode))
Pad.to_mlir_node = to_mlir_node
