import nnoir
import chainer
from nnoir_chainer import NNOIRFunction
import numpy as np
import util


def test_depthwise_convolution_2d():
    sy, sx = 2, 3
    ph, pw = 1, 3
    kh, kw = 3, 4
    batch = 2
    in_ch = 6
    in_h = 10
    in_w = 9
    ch_mul = 2
    out_ch = in_ch * ch_mul
    out_h = 5
    out_w = 4
    inputs = [nnoir.Value(b'v0', np.zeros((batch, in_ch, in_h, in_w)).astype('float32'))]
    outputs = [nnoir.Value(b'v1', np.zeros((batch, out_ch, out_h, out_w)).astype('float32'))]
    W = np.random.randn(ch_mul, in_ch, kh, kw).astype('float32')
    b = np.random.randn(out_ch).astype('float32')

    nodes = inputs + outputs
    input_names = [x.name for x in inputs]
    output_names = [x.name for x in outputs]
    function = nnoir.functions.DepthwiseConvolution2D(
        input_names, output_names, W=W, b=b, pad_h=(1, 1), pad_w=(3, 3), stride=(sy, sx), dilate=(1, 1))
    result = nnoir.NNOIR(b'DepthwiseConvolution2D', b'nnoir2chainer_test', '0.1', input_names, output_names, nodes, [function])
    result.dump('depthwise_convolution_2d.nnoir')

    x = np.random.randn(batch, in_ch, in_h, in_w).astype('float32')
    ref = function.run(x)
    with chainer.using_config('train', False):
        m = NNOIRFunction('depthwise_convolution_2d.nnoir')
        y = m(x)
        assert(np.all(abs(y-ref).data < util.epsilon))


def test_depthwise_convolution_2d_dilate():
    sy, sx = 2, 3
    ph, pw = 1, 3
    kh, kw = 3, 4
    batch = 2
    in_ch = 6
    in_h = 10
    in_w = 9
    ch_mul = 2
    out_ch = in_ch * ch_mul
    out_h = 5
    out_w = 4
    dy, dx = (2, 3)
    inputs = [nnoir.Value(b'v0', np.zeros((batch, in_ch, in_h, in_w)).astype('float32'))]
    outputs = [nnoir.Value(b'v1', np.zeros((batch, out_ch, out_h, out_w)).astype('float32'))]
    W = np.random.randn(ch_mul, in_ch, kh, kw).astype('float32')
    b = np.random.randn(out_ch).astype('float32')

    nodes = inputs + outputs
    input_names = [x.name for x in inputs]
    output_names = [x.name for x in outputs]
    function = nnoir.functions.DepthwiseConvolution2D(
        input_names, output_names, W=W, b=b, pad_h=(1, 1), pad_w=(3, 3), stride=(sy, sx), dilate=(dy, dx))
    result = nnoir.NNOIR(b'DepthwiseConvolution2D', b'nnoir2chainer_test', '0.1', input_names, output_names, nodes, [function])
    result.dump('depthwise_convolution_2d_dilate.nnoir')

    x = np.random.randn(batch, in_ch, in_h, in_w).astype('float32')
    ref = function.run(x)
    with chainer.using_config('train', False):
        m = NNOIRFunction('depthwise_convolution_2d_dilate.nnoir')
        y = m(x)
        assert(np.all(abs(y-ref).data < util.epsilon))
