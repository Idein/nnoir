#!/usr/bin/env python
import numpy as np

import chainer
from chainer import variable

import googlenet

import mlir.converter as C

# ./example.py > googlenet.mlir
if __name__ == '__main__':
    model = googlenet.GoogLeNet()
    model.train = False
    C.patch()
    x = variable.Variable(np.zeros((1,1,256,256)).astype(np.float32))
    t = variable.Variable(np.zeros(1).astype(np.int32))
    y = model(x, t)
    g = C.Chainer(model, (x,t), (y,))
    print (g.to_mlir())
