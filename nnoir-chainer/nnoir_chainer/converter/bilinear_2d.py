import chainer.functions as F


class ConvertBilinear2D():

    def __init__(self, function, inputs, outputs):
        if 'mode' not in function.params or function.params['mode'] == b'align_corners':
            self.f = lambda x: F.resize_images(x, tuple(function.params['size']))
        else:
            raise Exception('chainer has no correspond function Bilinear({})'.format(function.params['mode']))

    def __call__(self, x):
        return self.f(x)
