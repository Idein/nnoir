import mlir
import chainer
from mlir_chainer import MLIRFunction
import numpy as np
import util

def test_max_pooling_2d():
    kh, kw = 2, 3
    sy, sx = 1, 2
    pad_h, pad_w = 1, 2
    inputs  = [mlir.Value(b'v0', 'float', (2,3,4,5))]
    outputs = [mlir.Value(b'v2', 'float', (2,3,5,4))]
    nodes = inputs + outputs
    input_names = [ x.name for x in inputs ]
    output_names = [ x.name for x in outputs ]
    function = mlir.functions.MaxPooling2D(input_names, output_names,
                                       kernel=(kh, kw), stride=(sy, sx),
                                       pad_h=(pad_h, pad_h + sy - 1),
                                       pad_w=(pad_w, pad_w + sx - 1))
    result = mlir.MLIR(b'MaxPooling2D', b'mlir2chainer_test', 0.1, input_names, output_names, nodes, [function])
    result.dump('max_pooling_2d.mlir')

    x = np.random.randn(2,3,4,5)
    ref = function.run(x)
    m = MLIRFunction('max_pooling_2d.mlir')
    with chainer.using_config('train', False):
        y = m(x)
        assert(np.all(abs(y-ref).data<util.epsilon))
