# mlir-chainer

Chainer from/to MLIR converter

## Install

```
pip install .
```

## Example

### Import

```
import chainer
from mlir_chainer import MLIRFunction
m = MLIRFunction('mlir_file_path')
x = chainer.Variable(np_array)
with chainer.using_config('train', False):
    y = m(x)
    print(y)
```
