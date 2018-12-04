from .edge import Edge
import chainer.links as L

class Convolution2D(Edge):
    def __init__(self, inputs, outputs, **params):
        necessary_params = {'W',
                            'b',
                            'pad_h',
                            'pad_w',
                            'stride',
                            'dilate',
                            'groups'}
        optional_params = set()
        super().__init__(inputs, outputs, params, necessary_params, optional_params)
