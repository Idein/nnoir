import mlir
import chainer
import mlir2chainer
import numpy as np
import util

def test_linear_function():
    batch = 2
    in_ch = 3
    out_ch = 4
    inputs  = [mlir.Node('v0', 'float', (batch, in_ch)),
               mlir.Node('w0', 'float', (out_ch, in_ch)),
               mlir.Node('b0', 'float', (out_ch))]
    outputs = [mlir.Node('v2', 'float', (batch, out_ch))]
    nodes = inputs + outputs
    input_names = [ x.name for x in inputs ]
    output_names = [ x.name for x in outputs ]
    function = mlir.edges.LinearFunction(input_names, output_names)
    result = mlir.MLIR('LinearFunction', 'mlir2chainer_test', 0.1, input_names, output_names, nodes, [function])
    result.dump('linear_function.mlir')

    x = np.random.randn(batch, in_ch)
    W = np.random.randn(out_ch, in_ch)
    b = np.random.randn(out_ch)
    ref = function.run(x, W, b)
    m = mlir2chainer.ChainerNN('linear_function.mlir')
    with chainer.using_config('train', False):
        y = m(x, W, b)
        assert(np.all(abs(y-ref).data<util.epsilon))
