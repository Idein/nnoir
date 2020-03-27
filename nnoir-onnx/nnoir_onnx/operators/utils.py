import io
import numpy as np


class InvalidONNXData(Exception):

    def __init__(self, message):
        self.message = message


class UnsupportedONNXOperation(Exception):

    def __init__(self, node, message):
        self.node = node
        self.message = message


class Op:

    def __init__(self, node, opset_version):
        self.node = node
        self.opset_version = opset_version

    def to_function(self, env, constants):
        raise UnsupportedONNXOperation(self.node, "not implemented")


def encode_ndarray(obj):
    if obj is None:
        return None
    else:
        with io.BytesIO() as out:
            np.save(out, obj.copy())
            return {b'ndarray': out.getvalue()}


def auto_pad_to_manual_pad(n, k, s, d, auto_pad):
    dk = (k - 1) * d + 1
    if n % s == 0:
        pad = max(dk - s, 0)
    else:
        pad = max(dk - n % s, 0)
    if auto_pad == b'SAME_LOWER':
        pad_before = pad // 2
        pad_after = pad - pad_before
        return (pad_before, pad_after)
    elif auto_pad == b'SAME_UPPER':
        pad_after = pad // 2
        pad_before = pad - pad_after
        return (pad_before, pad_after)
    elif auto_pad == b'VALID':
        return (0, 0)
    else:
        raise 'invalid'


def gen_unregisterd_node_name(env):
    for i in range(len(env)):
        candidate = 'v{}'.format(i)
        if candidate not in env:
            return candidate


def register_node(env, name, val):
    env[name] = val
