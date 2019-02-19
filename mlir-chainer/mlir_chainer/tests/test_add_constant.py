import mlir
import chainer
from mlir_chainer import MLIRFunction
import numpy as np
import util


def test_add_constant():
    inputs = [mlir.Value(b'v0', np.zeros((10, 10)).astype('float32'))]
    outputs = [mlir.Value(b'v2', np.zeros((10, 10)).astype('float32'))]
    nodes = inputs + outputs
    input_names = [x.name for x in inputs]
    output_names = [x.name for x in outputs]
    function = mlir.functions.AddConstant(input_names, output_names, value=2.0)
    result = mlir.MLIR(b'AddConstant', b'mlir2chainer_test', '0.1', input_names, output_names, nodes, [function])
    result.dump('add_constant.mlir')

    x = np.random.randn(10, 10).astype('float32')
    ref = function.run(x)
    with chainer.using_config('train', False):
        m = MLIRFunction('add_constant.mlir')
        y = m(x)
        assert(np.all(abs(y-ref) < util.epsilon))
