from chainer.functions import AveragePooling2D
from chainer.mlir.patch import patched_function_apply, patched_function_call

if hasattr(AveragePooling2D, 'apply'):
    AveragePooling2D.apply = patched_function_apply(AveragePooling2D.apply)
else:
    AveragePooling2D.__call__ = patched_function_call(AveragePooling2D.__call__)

def to_mlir_node(self):
    return {
        b'name': 'AveragePooling2D',
        b'params': {
            b'kh': self.kh,
            b'kw': self.kw,
            b'sy': self.sy,
            b'sx': self.sx,
            b'ph': self.ph,
            b'pw': self.pw,
            b'cover_all': self.cover_all,
        }
    }
AveragePooling2D.to_mlir_node = to_mlir_node
