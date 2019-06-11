# nnoir-chainer

Chainer Model from/to NNOIR converter

## Install

```
pip install nnoir-chainer
```

## Example

### Import NNOIR

```
import chainer
from nnoir_chainer import NNOIRFunction
m = NNOIRFunction('nnoir_file_path')
x = chainer.Variable(np_array)
with chainer.using_config('train', False):
    y = m(x)
    print(y)
```

### Export NNOIR

```
m = model.CNN()
chainer.serializers.load_npz('cnn.model', L.Classifier(m))
with chainer.using_config('train', False):
    x = chainer.Variable(np.zeros((1, 28*28)).astype(np.float32))
    y = m(x)
    g = nnoir_chainer.Graph(m, (x,), (y,))
    result = g.to_nnoir()
    with open('model.nnoir', 'w') as f:
        f.buffer.write(result)
```

These layers are supported by nnoir-chainer exporter.

* chainer.links
    * BatchNormalization
    * Bias
    * Linear
    * Convolution2D (DepthwiseConvolution2D, DilatedConvolution2D)
    * Scale
    * Swish
* chainer.function
    * Add
    * AddConstant
    * AveragePooling2D
    * Concat
    * Dropout
    * ELU
    * LeakyReLU
    * MaxPooling2D
    * Mul
    * MulConstant
    * Pad
    * ReLU
    * Reshape
    * Sigmoid
    * Softmax
    * Tanh
    * Transpose
    * Unpooling2D
