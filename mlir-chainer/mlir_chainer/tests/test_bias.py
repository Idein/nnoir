import mlir
import chainer
from mlir_chainer import MLIRFunction
import numpy as np
import util


def test_bias():
    inputs = [mlir.Value(b'v0', np.zeros((2, 3, 4, 5)).astype('float32'))]
    outputs = [mlir.Value(b'v2', np.zeros((2, 3, 4, 5)).astype('float32'))]
    nodes = inputs + outputs
    input_names = [x.name for x in inputs]
    output_names = [x.name for x in outputs]
    b = np.random.randn(3).astype('float32')
    params = {'axis': 1,
              'b': b}
    function = mlir.functions.Bias(input_names, output_names, **params)
    result = mlir.MLIR(b'Bias', b'mlir2chainer_test', '0.1', input_names, output_names, nodes, [function])
    result.dump('bias.mlir')

    x = np.random.randn(2, 3, 4, 5).astype('float32')
    ref = function.run(x)
    with chainer.using_config('train', False):
        m = MLIRFunction('bias.mlir')
        y = m(x)
        assert(np.all(abs(y-ref).data < util.epsilon))
