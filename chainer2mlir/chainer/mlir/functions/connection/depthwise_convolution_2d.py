from chainer.functions.connection.depthwise_convolution_2d import DepthwiseConvolution2D
from chainer.mlir.patch import patched_function_apply, patched_function_call

if hasattr(DepthwiseConvolution2D, 'apply'):
    DepthwiseConvolution2D.apply = patched_function_apply(DepthwiseConvolution2D.apply)
else:
    DepthwiseConvolution2D.__call__ = patched_function_call(DepthwiseConvolution2D.__call__)

def to_mlir_node(self):
    return {
        b'name': 'DepthwiseConvolution2D',
        b'params': {}
    }
DepthwiseConvolution2D.to_mlir_node = to_mlir_node
