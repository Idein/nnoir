import mlir
import chainer
from mlir_chainer import MLIRFunction
import numpy as np
import util

def test_local_response_normalization():
    inputs  = [mlir.Value(b'v0', 'float', (10,10))]
    outputs = [mlir.Value(b'v2', 'float', (10,10))]
    nodes = inputs + outputs
    input_names = [ x.name for x in inputs ]
    output_names = [ x.name for x in outputs ]
    function = mlir.functions.LocalResponseNormalization(input_names, output_names, n=6, k=3, alpha=0.0002, beta=0.8)
    result = mlir.MLIR(b'LocalResponseNormalization', b'mlir2chainer_test', 0.1, input_names, output_names, nodes, [function])
    result.dump('local_response_normalization.mlir')

    x = np.random.randn(10,10)
    ref = function.run(x)
    m = MLIRFunction('local_response_normalization.mlir')
    with chainer.using_config('train', False):
        y = m(x)
        assert(np.all(abs(y-ref).data<util.epsilon))
