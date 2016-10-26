#!/usr/bin/env python
import sys
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
    result = g.to_mlir()
    if sys.version_info < (3, 0, 0):
        print(result)
    else:
        sys.stdout.buffer.write(result)
