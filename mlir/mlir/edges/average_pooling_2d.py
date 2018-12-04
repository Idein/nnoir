from .edge import Edge
import chainer.functions as F

class AveragePooling2D(Edge):
    def __init__(self, inputs, outputs, **params):
        necessary_params = {'kernel',
                            'stride',
                            'pad_h',
                            'pad_w'}
        optional_params = set()
        super().__init__(inputs, outputs, params, necessary_params, optional_params)
