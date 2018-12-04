import chainer.functions as F

class ConvertUnpooling2D():
    def to_chainer(edge, x):
        return F.unpooling_2d(x,
                              ksize = (edge.params["kh"], edge.params["kw"]),
                              stride = (edge.params["sy"], edge.params["sx"]),
                              pad = (edge.params["ph"], edge.params["pw"]),
                              outsize = (edge.params["outh"], edge.params["outw"]),
                              cover_all = edge.params["cover_all"])
