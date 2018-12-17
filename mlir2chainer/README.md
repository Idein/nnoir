# mlir2chainer

## install
` pip install .`

## example
```
import chainer
from chainer.links.mlir import MLIRFunction
m = MLIRFunction('mlir_file_path')
x = chainer.Variable(np_array)
with chainer.using_config('train', False):
    y = m(x)
    print(y)
```
