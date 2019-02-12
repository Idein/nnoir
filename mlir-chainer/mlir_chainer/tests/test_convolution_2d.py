import mlir
import chainer
from mlir_chainer import MLIRFunction
import numpy as np
import util

def test_convolution_2d():
    batch = 2
    in_ch = 4
    in_h = 10
    in_w = 9
    out_ch = 7
    out_h = 6
    out_w = 3
    kh = 4
    kw = 3
    inputs  = [mlir.Value(b'v0', 'float', (batch, in_ch, in_h, in_w))]
    outputs = [mlir.Value(b'v1', 'float', (batch,out_ch,out_h, out_w))]
    W = np.random.randn(out_ch, in_ch, kh, kw).astype(np.float32)
    b = np.random.randn(out_ch).astype(np.float32)

    nodes = inputs + outputs
    input_names = [ x.name for x in inputs ]
    output_names = [ x.name for x in outputs ]
    function = mlir.functions.Convolution2D(input_names, output_names, W=W, b=b, pad_h=(2,2), pad_w=(1,1), stride=(2,3), dilate=(1,1), groups=1)
    result = mlir.MLIR(b'Convolution2D', b'mlir2chainer_test', 0.1, input_names, output_names, nodes, [function])
    result.dump('convolution_2d.mlir')

    x = np.random.randn(batch, in_ch, in_h, in_w).astype(np.float32)
    ref = function.run(x)
    m = MLIRFunction('convolution_2d.mlir')
    with chainer.using_config('train', False):
        y = m(x)
        assert(np.all(abs(y-ref).data<util.epsilon))
