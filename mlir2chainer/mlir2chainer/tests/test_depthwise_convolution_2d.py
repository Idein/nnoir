import mlir
import chainer
import mlir2chainer
import numpy as np
import six
import msgpack
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
    inputs  = [mlir.Node('v0', 'float', (batch, in_ch, in_h, in_w))]
    outputs = [mlir.Node('v1', 'float', (batch,out_ch,out_h, out_w))]
    W = np.arange(ch_mul*in_ch*kh*kw).reshape(ch_mul, in_ch, kh, kw).astype('float32')
    b = np.arange(out_ch).astype('float32')
        
    nodes = inputs + outputs
    input_names = [ x.name for x in inputs ]
    output_names = [ x.name for x in outputs ]
    function = mlir.edges.DepthwiseConvolution2D(input_names, output_names, W=W, b=b, pad_h=(1,1), pad_w=(3,3), stride=(sy, sx), dilate=(1,1))
    result = mlir.MLIR('DepthwiseConvolution2D', 'mlir2chainer_test', 0.1, input_names, output_names, nodes, [function])
    result.dump('depthwise_convolution_2d.mlir')

    x = np.random.randn(batch, in_ch, in_h, in_w)
    ref = function.run(x)
    m = mlir2chainer.ChainerNN('depthwise_convolution_2d.mlir')
    with chainer.using_config('train', False):
        y = m(x)
        assert(np.all(abs(y-ref).data<util.epsilon))

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
    inputs  = [mlir.Node('v0', 'float', (batch, in_ch, in_h, in_w))]
    outputs = [mlir.Node('v1', 'float', (batch,out_ch,out_h, out_w))]
    W = np.arange(ch_mul*in_ch*kh*kw).reshape(ch_mul, in_ch, kh, kw).astype('float32')
    b = np.arange(out_ch).astype('float32')
        
    nodes = inputs + outputs
    input_names = [ x.name for x in inputs ]
    output_names = [ x.name for x in outputs ]
    function = mlir.edges.DepthwiseConvolution2D(input_names, output_names, W=W, b=b, pad_h=(1,1), pad_w=(3,3), stride=(sy, sx), dilate=(dy,dx))
    result = mlir.MLIR('DepthwiseConvolution2D', 'mlir2chainer_test', 0.1, input_names, output_names, nodes, [function])
    result.dump('depthwise_convolution_2d.mlir')

    x = np.random.randn(batch, in_ch, in_h, in_w)
    ref = function.run(x)
    m = mlir2chainer.ChainerNN('depthwise_convolution_2d.mlir')
    with chainer.using_config('train', False):
        y = m(x)
        assert(np.all(abs(y-ref).data<util.epsilon))
