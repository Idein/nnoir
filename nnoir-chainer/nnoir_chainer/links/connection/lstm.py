from chainer.links import LSTM
from nnoir_chainer.patch import encode_ndarray, patched_link_call
import numpy as np
import nnoir.functions as NNOIR

LSTM.__call__ = patched_link_call(LSTM.__call__)


def to_nnoir_node(self, inputs, outputs):
    return NNOIR.LSTM(
        inputs,
        outputs,
    )


LSTM.to_nnoir_node = to_nnoir_node
