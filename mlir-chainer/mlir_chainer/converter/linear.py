import chainer
import chainer.links as L


class ConvertLinear(chainer.Chain):

    def __init__(self, function, inputs, outputs):
        super(ConvertLinear, self).__init__()
        with self.init_scope():
            self.fc = L.Linear(in_size=function.params['W'].shape[1],
                               out_size=function.params['W'].shape[0],
                               nobias=(function.params['b'] is None),
                               initialW=function.params['W'],
                               initial_bias=function.params['b'])

    def __call__(self, x):
        return self.fc(x)
