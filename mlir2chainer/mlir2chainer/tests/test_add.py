import mlir
import chainer
import mlir2chainer
import numpy as np
import six
import msgpack

def test_add():
    # mlirを作成
    inputs  = [mlir.Node('v0', 'float', (10,10)),
               mlir.Node('v1', 'float', (10,10))]
    outputs = [mlir.Node('v2', 'float', (10,10))]
    nodes = inputs + outputs
    input_names = [ x.name for x in inputs ]
    output_names = [ x.name for x in outputs ]
    function = mlir.edges.Add(input_names, output_names)
    result = mlir.MLIR('Add', 'mlir2chainer_test', 0.1, input_names, output_names, nodes, [function])
    result.dump('add.mlir')
    
    # mlirをload
    m = mlir2chainer.ChainerNN('add.mlir')

    # 実行
    x1 = np.zeros((10,10))
    x2 = np.zeros((10,10))
    x1[:] = 1
    x2[:] = 2
    with chainer.using_config('train', False):
        y = m(x1, x2)
        assert(np.all(y.data==3))
