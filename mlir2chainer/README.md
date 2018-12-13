# mlir2chainer

## install
` pip install .`

## example
```
import mlir2chainer
m = mlir2chainer.ChainerNN('mlir_file_path')
x = chainer.Variable(np_array)
with chainer.using_config('train', False):
    y = m(x)
    print(y)
```