import chainer
import chainer.functions as F
import chainer.links as L


class CNN(chainer.Chain):
    def __init__(self):
        super(CNN, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 16, 5)
            self.conv2 = L.Convolution2D(None, 32, 5)
            self.l1 = L.Linear(None, 32)
            self.l2 = L.Linear(None, 10)

    def __call__(self, x):
        h1 = F.max_pooling_2d(F.relu(self.conv1(F.reshape(x, (-1, 1, 28, 28)))), 2)
        h2 = F.max_pooling_2d(F.relu(self.conv2(h1)), 2)
        h3 = F.relu(self.l1(h2))
        return self.l2(h3)
