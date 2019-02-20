# nnoir-chainer

Chainer from/to NNOIR converter

## Install

```
pip install .
```

## Example

### Import

```
import chainer
from nnoir_chainer import NNOIRFunction
m = NNOIRFunction('nnoir_file_path')
x = chainer.Variable(np_array)
with chainer.using_config('train', False):
    y = m(x)
    print(y)
```
