import chainer
import chainer.links as L

class ConvertLinear(chainer.Chain):

    def __init__(self, edge, inputs, outputs):
        super(ConvertLinear, self).__init__()
        with self.init_scope():
            self.fc = L.Linear(in_size = edge.params['W'].shape[1],
                               out_size = edge.params['W'].shape[0],
                               nobias = (edge.params['b'] is None),
                               initialW = edge.params['W'],
                               initial_bias = edge.params['b'])

    def __call__(self, x):
        return self.fc(x)
