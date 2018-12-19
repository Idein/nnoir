from chainer.functions.array.resize_images import ResizeImages
from mlir_chainer.patch import patched_function_apply, patched_function_call

if hasattr(ResizeImages, 'apply'):
    ResizeImages.apply = patched_function_apply(ResizeImages.apply)
else:
    ResizeImages.__call__ = patched_function_call(ResizeImages.__call__)

def to_mlir_node(self):
    return {
        b'name': 'Bilinear2D',
        b'params': {
            b'size': (self.out_H, self.out_W)
        }
    }
ResizeImages.to_mlir_node = to_mlir_node
