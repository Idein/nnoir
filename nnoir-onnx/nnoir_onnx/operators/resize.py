from nnoir.functions import *
from .utils import *


class OpResize(Op):

    def __init__(self, node, *args):
        super(OpResize, self).__init__(node, *args)

        if self.opset_version < 11:
            raise UnsupportedONNXOperation(self.node, 'only opset_version >= 11 is supported')

        self.mode = b'half_pixel'  # unsupported default ONNX coordinate_transformation_mode
        for attr in node.attribute:
            if attr.name == 'mode':
                if attr.s != b'linear':
                    raise UnsupportedONNXOperation(self.node, '{} mode is not supported for Resize, '
                                                              'only linear.'.format(attr.s.decode('utf-8')))
            if attr.name == 'coordinate_transformation_mode':
                if attr.s == b'pytorch_half_pixel':
                    self.mode = b'align_centers'
                elif attr.s == b'align_corners':
                    self.mode = b'align_corners'
                else:
                    raise UnsupportedONNXOperation(
                        self.node, '{} coordinate_transformation_mode '
                        'is not supported for Resize, only '
                        '"align_corners" and "pytorch_half_pixel"'.format(attr.s.decode('utf-8')))

        if self.mode == b'half_pixel':
            raise UnsupportedONNXOperation(
                self.node, 'default "half_pixel" coordinate_transformation_mode is unsupported')

    def to_function(self, env, constants):
        x, *_ = self.node.input
        [y] = self.node.output
        return [
            Bilinear2D(
                [x],
                list(self.node.output),
                size=tuple(env[y].shape[2:]),
                mode=self.mode
            )
        ]
