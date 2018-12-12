import chainer.functions as F

class ConvertAveragePooling2D():
    def to_chainer(self, edge, x):
        sy, sx = edge.params['stride']
        if edge.params['pad_h'][0] + sy - 1 != edge.params['pad_h'][1] or edge.params['pad_w'][0] + sx - 1 != edge.params['pad_w'][1]:
            raise Exception('this padding is not supported now')
        return F.average_pooling_2d(x,
                                    ksize = tuple(edge.params['kernel']),
                                    stride = (sy, sx),
                                    pad = (edge.params['pad_h'][0], edge.params['pad_w'][0]))
