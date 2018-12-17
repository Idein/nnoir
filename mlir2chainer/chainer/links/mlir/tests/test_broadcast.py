import mlir
import chainer
from chainer.links.mlir import MLIRFunction
import numpy as np
import sys
import chainer.functions as F
import util

def test_broadcast():
    in_shape = (1,1,4,5)
    out_shape = (2,3,4,5)
    inputs  = [mlir.Node('v0', 'float', in_shape)]
    outputs = [mlir.Node('v2', 'float', out_shape)]
    nodes = inputs + outputs
    input_names = [ x.name for x in inputs ]
    output_names = [ x.name for x in outputs ]
    function = mlir.edges.BroadcastTo(input_names, output_names)

    result = mlir.MLIR('BroadcastTo', 'mlir2chainer_test', 0.1, input_names, output_names, nodes, [function])
    result.dump('broadcast.mlir')

    x = np.random.randn(*in_shape)
    ref = function.run(x, out_shape)
    with chainer.using_config('train', False):
        m = MLIRFunction('broadcast.mlir')
        y = m(x)
        assert(np.all(abs(y-ref).data<util.epsilon))
