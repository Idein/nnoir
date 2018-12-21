from chainer.functions.array.resize_images import ResizeImages
from mlir_chainer.patch import patched_function_apply, patched_function_call
import mlir.edges as MLIR

if hasattr(ResizeImages, 'apply'):
    ResizeImages.apply = patched_function_apply(ResizeImages.apply)
else:
    ResizeImages.__call__ = patched_function_call(ResizeImages.__call__)

def to_mlir_node(self, inputs, outputs):
    return MLIR.Bilinear2D(inputs, outputs, size=(self.out_H, self.out_W))
ResizeImages.to_mlir_node = to_mlir_node
