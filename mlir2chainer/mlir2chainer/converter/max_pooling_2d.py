import chainer.functions as F

class ConvertMaxPooling2D():
    def to_chainer(edge, x):
        sy, sx = tuple(edge.params['stride'])
        if edge.params['pad_h'][0] + sy - 1 != edge.params['pad_h'][1] or edge.params['pad_w'][0] + sx - 1 != edge.params['pad_w'][1]:
            # ここである条件を満たせばcover_all=Trueで実行できる？
            raise Exception('this padding is not supported now')
        return F.max_pooling_2d(x,
                                ksize = edge.params['kernel'],
                                stride = (sy, sx),
                                pad = (edge.params['pad_h'][0], edge.params['pad_w'][0]))
