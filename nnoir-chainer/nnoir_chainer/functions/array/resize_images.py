from chainer.functions.array.resize_images import ResizeImages
from nnoir_chainer.patch import patched_function_apply, patched_function_call
import nnoir.functions as NNOIR

if hasattr(ResizeImages, 'apply'):
    ResizeImages.apply = patched_function_apply(ResizeImages.apply)
else:
    ResizeImages.__call__ = patched_function_call(ResizeImages.__call__)


def to_nnoir_node(self, inputs, outputs):
    return NNOIR.Bilinear2D([x.name for x in inputs], [x.name for x in outputs], size=(self.out_H, self.out_W), mode='align_corners')


ResizeImages.to_nnoir_node = to_nnoir_node
