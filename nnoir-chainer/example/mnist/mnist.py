#!/usr/bin/env python
# -*- coding: utf-8 -*-
import chainer
import chainer.links as L
import model
import nnoir_chainer
import numpy as np

m = model.CNN()
chainer.serializers.load_npz("cnn.model", L.Classifier(m))

with chainer.using_config("train", False):
    x = chainer.Variable(np.zeros((1, 28 * 28)).astype(np.float32))
    y = m(x)
    g = nnoir_chainer.Graph(m, (x,), (y,))
    result = g.to_nnoir()
    with open("model.nnoir", "w") as f:
        f.buffer.write(result)
