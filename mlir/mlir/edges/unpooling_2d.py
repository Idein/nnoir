import chainer.functions as F
from .edge import Edge

class Unpooling2D(Edge):
    def __init__(self, inputs, outputs, **params):
        necessary_params = {'kh',
                            'kw',
                            'sy',
                            'sx',
                            'ph',
                            'pw',
                            'cover_all',
                            'outh',
                            'outw'}
        optional_params = set()
        super().__init__(inputs, outputs, params, necessary_params, optional_params)
