import mlir
import chainer
from chainer.links.mlir import MLIRFunction
import numpy as np
import util

def test_local_response_normalization():
    inputs  = [mlir.Node('v0', 'float', (10,10))]
    outputs = [mlir.Node('v2', 'float', (10,10))]
    nodes = inputs + outputs
    input_names = [ x.name for x in inputs ]
    output_names = [ x.name for x in outputs ]
    function = mlir.edges.LocalResponseNormalization(input_names, output_names, n=6, k=3, alpha=0.0002, beta=0.8)
    result = mlir.MLIR('LocalResponseNormalization', 'mlir2chainer_test', 0.1, input_names, output_names, nodes, [function])
    result.dump('local_response_normalization.mlir')

    x = np.random.randn(10,10)
    ref = function.run(x)
    m = MLIRFunction('local_response_normalization.mlir')
    with chainer.using_config('train', False):
        y = m(x)
        assert(np.all(abs(y-ref).data<util.epsilon))
