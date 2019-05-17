import nnoir
import chainer
from nnoir_chainer import NNOIRFunction, Graph
import numpy as np
import util


def test_import_add_constant():
    inputs = [nnoir.Value(b'v0', np.zeros((10, 10)).astype('float32'))]
    outputs = [nnoir.Value(b'v2', np.zeros((10, 10)).astype('float32'))]
    nodes = inputs + outputs
    input_names = [x.name for x in inputs]
    output_names = [x.name for x in outputs]
    function = nnoir.functions.AddConstant(input_names, output_names, value=2.0)
    result = nnoir.NNOIR(b'AddConstant', b'nnoir2chainer_test', '0.1', input_names, output_names, nodes, [function])
    result.dump('import_add_constant.nnoir')

    x = np.random.randn(10, 10).astype('float32')
    ref = function.run(x)
    with chainer.using_config('train', False):
        m = NNOIRFunction('import_add_constant.nnoir')
        y = m(x)
        assert(np.all(abs(y-ref) < util.epsilon))


def test_export_add_constant():

    class Model(chainer.Link):
        def __call__(self, x):
            return x + 1.0

    with chainer.using_config('train', False):
        model = Model()
        x = chainer.Variable(np.array([0, 1, 2]).astype(np.float32))
        y = model(x)
        g = Graph(model, (x,), (y,))
        result = g.to_nnoir()
        with open('export_add_constant.nnoir', 'w') as f:
            f.buffer.write(result)

        m = NNOIRFunction('export_add_constant.nnoir')
        z = m(x)
        assert(np.all(abs(z.data-y.data) < util.epsilon))
