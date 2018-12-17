import chainer.links as L

class ConvertLinear():

    def __init__(self, edge, inputs, outputs):
        self.fc = L.Linear(in_size = edge.params['W'].shape[1],
                           out_size = edge.params['W'].shape[0],
                           nobias = (edge.params['b'] is None),
                           initialW = edge.params['W'],
                           initial_bias = edge.params['b'])

    def __call__(self, x):
        return self.fc(x)
