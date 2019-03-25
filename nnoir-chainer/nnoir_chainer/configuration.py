from chainer.configuration import LocalConfig


def _patch__getattr__(self, name):
    if name == 'enable_backprop':
        return True
    else:
        if hasattr(self._local, name):
            return getattr(self._local, name)
        return getattr(self._global, name)


def _patch__setattr__(self, name, value):
    if name == 'enable_backprop' and not value:
        raise Exception('nnoir_chainer force enable_backprop')
    setattr(self._local, name, value)


LocalConfig.__getattr__ = _patch__getattr__
LocalConfig.__setattr__ = _patch__setattr__
