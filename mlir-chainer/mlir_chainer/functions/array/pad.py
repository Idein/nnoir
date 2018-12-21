from chainer.functions.array.pad import Pad
from mlir_chainer.patch import patched_function_apply, patched_function_call
from numpy.lib.arraypad import _validate_lengths # dirty
import mlir.edges as MLIR

if hasattr(Pad, 'apply'):
    Pad.apply = patched_function_apply(Pad.apply)
else:
    Pad.__call__ = patched_function_call(Pad.__call__)

def to_mlir_node(self, inputs, outputs):
    [var] = self.chainer_input_variables
    if self.mode == 'constant':
        if type(self.keywords['constant_values']) == int:
            value = float(self.keywords['constant_values'])
        elif type(self.keywords['constant_values'][0]) == tuple:
            value = float(self.keywords['constant_values'][0][0])
            for (b,a) in self.keywords['constant_values']:
                if b != value or a != value:
                    raise Exception('unsupported pad value "{}"'.format(self.keywords['constant_values']))
        else:
            value = float(self.keywords['constant_values'][0])
        return MLIR.ConstantPadding(
            inputs,
            outputs,
            pads=_validate_lengths(var.data, self.pad_width),
            value=value
        )
    else:
        raise Exception('unsupported pad mode "{}"'.format(self.mode))
Pad.to_mlir_node = to_mlir_node
