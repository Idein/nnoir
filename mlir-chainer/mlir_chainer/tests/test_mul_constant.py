import mlir
import chainer
from mlir_chainer import MLIRFunction
import numpy as np
import util

def test_mul_constant():
    inputs  = [mlir.Value(b'v0', 'float', (10,10))]
    outputs = [mlir.Value(b'v2', 'float', (10,10))]
    nodes = inputs + outputs
    input_names = [ x.name for x in inputs ]
    output_names = [ x.name for x in outputs ]
    function = mlir.functions.MulConstant(input_names, output_names, value=2)
    result = mlir.MLIR(b'MulConstant', b'mlir2chainer_test', 0.1, input_names, output_names, nodes, [function])
    result.dump('mul_constant.mlir')

    x = np.random.randn(10,10)
    ref = function.run(x)
    m = MLIRFunction('mul_constant.mlir')
    with chainer.using_config('train', False):
        y = m(x)
        assert(np.all(abs(y-ref)<util.epsilon))
