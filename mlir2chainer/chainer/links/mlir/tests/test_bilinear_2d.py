import mlir
import chainer
from chainer.links.mlir import MLIRFunction
import numpy as np
import sys
import chainer.functions as F
import util

def test_bilinear_2d():
    in_shape = (2,3,9,10)
    out_shape = (2,3,4,5)
    inputs  = [mlir.Node('v0', 'float', in_shape)]
    outputs = [mlir.Node('v2', 'float', out_shape)]
    nodes = inputs + outputs
    input_names = [ x.name for x in inputs ]
    output_names = [ x.name for x in outputs ]
    function = mlir.edges.Bilinear2D(input_names, output_names, size=(out_shape[2], out_shape[3]))

    result = mlir.MLIR('Bilinear2D', 'mlir2chainer_test', 0.1, input_names, output_names, nodes, [function])
    result.dump('bilinear_2d.mlir')

    x = np.random.randn(*in_shape).astype(np.float32)
    ref = function.run(x)
    with chainer.using_config('train', False):
        m = MLIRFunction('bilinear_2d.mlir')
        y = m(x)
    assert(np.all(abs(y-ref).data<util.epsilon))
