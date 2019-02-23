from chainer.functions.array.pad import Pad
from nnoir_chainer.patch import patched_function_apply, patched_function_call
import nnoir.functions as NNOIR
import numpy as np

if hasattr(Pad, 'apply'):
    Pad.apply = patched_function_apply(Pad.apply)
else:
    Pad.__call__ = patched_function_call(Pad.__call__)


def to_nnoir_node(self, inputs, outputs):
    [var] = self.chainer_input_variables
    zero_padded_ones = np.pad(np.ones(var.shape, dtype=np.int32), self.pad_width, mode='constant', constant_values=0)
    ones_positions = np.transpose(np.where(zero_padded_ones > 0))
    pad_befores = ones_positions[0]
    pad_afters = np.array(zero_padded_ones.shape) - ones_positions[-1] - 1
    pad_width = list(map(lambda x: (int(x[0]), int(x[1])), np.stack([pad_befores, pad_afters]).T))
    if self.mode == 'constant':
        if type(self.keywords['constant_values']) == int:
            value = float(self.keywords['constant_values'])
        elif type(self.keywords['constant_values'][0]) == tuple:
            value = float(self.keywords['constant_values'][0][0])
            for (b, a) in self.keywords['constant_values']:
                if b != value or a != value:
                    raise Exception('unsupported pad value "{}"'.format(self.keywords['constant_values']))
        else:
            value = float(self.keywords['constant_values'][0])
        return NNOIR.ConstantPadding(
            [x.name for x in inputs],
            [x.name for x in outputs],
            pads=pad_width,
            value=value
        )
    else:
        raise Exception('unsupported pad mode "{}"'.format(self.mode))


Pad.to_nnoir_node = to_nnoir_node
