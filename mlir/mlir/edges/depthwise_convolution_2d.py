import numpy
import six
from .edge import Edge
import chainer.links as L

class DepthwiseConvolution2D(Edge):
    def __init__(self, inputs, outputs, **params):
        necessary_params = {'W',
                            'b',
                            'stride',
                            'pad_h',
                            'pad_w',
                            'dilate'}
        optional_params = set()
        super().__init__(inputs, outputs, params, necessary_params, optional_params)
