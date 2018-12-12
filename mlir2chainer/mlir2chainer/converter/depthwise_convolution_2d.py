import numpy
import six
import chainer.links as L

class ConvertDepthwiseConvolution2D():
    def to_chainer(edge, x):
        if edge.params["pad_h"][0] != edge.params["pad_h"][1] or edge.params["pad_w"][0] != edge.params["pad_w"][1]:
            raise Exception('this padding is not supported now')
        if tuple(edge.params["dilate"]) == (1,1): #depthwise
            conv = L.DepthwiseConvolution2D(in_channels = edge.params["W"].shape[1],
                                            channel_multiplier = edge.params["W"].shape[0],
                                            ksize = (edge.params["W"].shape[2], edge.params["W"].shape[3]),
                                            stride = tuple(edge.params["stride"]),
                                            pad = (edge.params["pad_h"][0], edge.params["pad_w"][0]),
                                            nobias = (edge.params["b"] is None))
            conv.W.data = edge.params["W"]
        else:
            in_channels = x.shape[1]
            groups = edge.params["W"].shape[1]
            out_channels = edge.params["W"].shape[0] * groups
            kh = edge.params["W"].shape[2]
            kw = edge.params["W"].shape[3]
            conv = L.Convolution2D(in_channels = groups,
                                   out_channels = out_channels,
                                   ksize = (kh, kw),
                                   stride = tuple(edge.params["stride"]),
                                   pad = (edge.params["pad_h"][0], edge.params["pad_w"][0]),
                                   nobias = (edge.params["b"] is None),
                                   dilate = edge.params["dilate"],
                                   groups = groups)
            conv.W.data = numpy.rollaxis(edge.params["W"], 1, 0).reshape(out_channels, in_channels//groups, kh, kw)
        if edge.params["b"] is not None:
            conv.b.data = edge.params["b"]
        return conv(x)
