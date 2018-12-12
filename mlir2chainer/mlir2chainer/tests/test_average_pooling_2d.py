import mlir
import chainer
import mlir2chainer
import numpy as np
import six
import msgpack
import util
import chainer.functions as F

def test_average_pooling_2d():
    ksize = (2,3)
    stride = (1,2)
    pad = (0,1)

    inputs  = [mlir.Node('v0', 'float', (2,3,4,5))]
    outputs = [mlir.Node('v2', 'float', (2,3,3,3))]
    nodes = inputs + outputs
    input_names = [ x.name for x in inputs ]
    output_names = [ x.name for x in outputs ]
    function = mlir.edges.AveragePooling2D(input_names, output_names,
                                           kernel=list(ksize),
                                           stride=list(stride),
                                           pad_h=[pad[0],pad[0]+stride[0]-1],
                                           pad_w=[pad[1],pad[1]+stride[1]-1])
    result = mlir.MLIR('AveragePooling2D', 'mlir2chainer_test', 0.1, input_names, output_names, nodes, [function])
    result.dump('average_pooling_2d.mlir')
    
    x = np.random.randn(2,3,4,5).astype(np.float32)
    ref = function.run(x)
    with chainer.using_config('train', False):
        m = mlir2chainer.ChainerNN('average_pooling_2d.mlir')
        y = m(x)
        assert(np.all(abs(y-ref).data<util.epsilon))
