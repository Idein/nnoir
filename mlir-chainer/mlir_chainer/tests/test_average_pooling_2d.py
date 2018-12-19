import mlir
import chainer
from chainer.links.mlir import MLIRFunction
import numpy as np
import util
import chainer.functions as F

def test_average_pooling_2d():
    ksize = (2,3)
    stride = (1,2)
    pad = (0,1)

    inputs  = [mlir.Node(b'v0', 'float', (2,3,4,5))]
    outputs = [mlir.Node(b'v2', 'float', (2,3,3,3))]
    nodes = inputs + outputs
    input_names = [ x.name for x in inputs ]
    output_names = [ x.name for x in outputs ]
    function = mlir.edges.AveragePooling2D(input_names, output_names,
                                           kernel=list(ksize),
                                           stride=list(stride),
                                           pad_h=[pad[0],pad[0]+stride[0]-1],
                                           pad_w=[pad[1],pad[1]+stride[1]-1],
                                           count_exclude_pad=False)
    result = mlir.MLIR(b'AveragePooling2D', b'mlir2chainer_test', 0.1, input_names, output_names, nodes, [function])
    result.dump('average_pooling_2d.mlir')

    x = np.random.randn(2,3,4,5)
    ref = function.run(x)
    with chainer.using_config('train', False):
        m = MLIRFunction('average_pooling_2d.mlir')
        y = m(x)
        assert(np.all(abs(y-ref).data<util.epsilon))
