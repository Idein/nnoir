import mlir
import chainer
from mlir_chainer import MLIRFunction
import numpy as np
import util

def test_bias():
    inputs  = [mlir.Value(b'v0', 'float', (2,3,4,5))]
    outputs = [mlir.Value(b'v2', 'float', (2,3,4,5))]
    nodes = inputs + outputs
    input_names = [ x.name for x in inputs ]
    output_names = [ x.name for x in outputs ]
    b = np.random.randn(10)
    params={'axis': 1,
            'b' : b}
    function = mlir.functions.Bias(input_names, output_names, **params)
    result = mlir.MLIR(b'Bias', b'mlir2chainer_test', 0.1, input_names, output_names, nodes, [function])
    result.dump('bias.mlir')

    x = np.random.randn(10,10)
    ref = function.run(x)
    m = MLIRFunction('bias.mlir')
    with chainer.using_config('train', False):
        y = m(x)
        assert(np.all(abs(y-ref).data<util.epsilon))
