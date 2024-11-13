# nnoir-onnx

nnoir-onnx is a converter from ONNX model to NNOIR model.

## Install
From [PyPI](https://pypi.org/project/nnoir-onnx/):

```
pip install nnoir-onnx
```

From [Dockerhub](https://hub.docker.com/repository/docker/idein/nnoir-tools):

```
docker pull idein/nnoir-tools:20240208
```

## Example

~~~~bash
wget https://www.cntk.ai/OnnxModels/mnist/opset_7/mnist.tar.gz
tar xvzf mnist.tar.gz
onnx2nnoir -o model.nnoir mnist/model.onnx
~~~~

With docker:

```
docker run --rm -it -u $UID:$GID -v $(pwd):/work idein/nnoir-tools:20240208 onnx2nnoir --graph_name "mobilenet" -o mobilenetv2-1.0.nnoir mobilenetv2-1.0.onnx
```

## Supported ONNX Operators

* [Add](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Add)
* [AveragePool](https://github.com/onnx/onnx/blob/master/docs/Operators.md#AveragePool)
* [BatchNormalization](https://github.com/onnx/onnx/blob/master/docs/Operators.md#BatchNormalization)
    * `scale`, `B`, `mean`, and `var` must be `"constant"`
* [Clip](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Clip)
    * must be opset version 6 or 11
    * if opset version is 11
      * `max` must be `"constant"`
    * `min` must be `0`
* [Concat](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Concat)
* [Conv](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Conv)
    * `W` must be [Constant](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Constant) value or have initializer value
    * `b` must be [Constant](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Constant) value or have initializer value
* [Cos](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Cos)
* [Div](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Div)
    * 1st input must not be `"constant"`
* [Dropout](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Dropout)
    * equivalent identity function
* [Elu](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Elu)
* [Exp](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Exp)
* [Flatten](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Flatten)
* [Gemm](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Gemm)
    * `B` must be [Constant](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Constant) value or have initializer value
    * `C` must be [Constant](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Constant) value or have initializer value
* [GlobalAveragePool](https://github.com/onnx/onnx/blob/master/docs/Operators.md#GlobalAveragePool)
* [HardSigmoid](https://github.com/onnx/onnx/blob/main/docs/Operators.md#hardsigmoid)
* [HardSwish](https://github.com/onnx/onnx/blob/main/docs/Operators.md#hardswish)
* [LeakyRelu](https://github.com/onnx/onnx/blob/master/docs/Operators.md#LeakyRelu)
* [LRN](https://github.com/onnx/onnx/blob/master/docs/Operators.md#LRN)
* [LSTM](https://github.com/onnx/onnx/blob/master/docs/Operators.md#lstm)
    * only `seq_length == 1`
    * `direction` must be forward
    * Supported `activations` are below
        * `Sigmoid`
        * `Tanh`
        * `Relu`
    * Not support `clip` and `input_forget`
* [MatMul](https://github.com/onnx/onnx/blob/master/docs/Operators.md#MatMul)
* [MaxPool](https://github.com/onnx/onnx/blob/master/docs/Operators.md#MaxPool)
    * `ceil_mode = 1` is not supported
    * `dilations` must be array of 1.
* [Mul](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Mul)
* [Pad](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Pad)
    * `mode` must be `"constant"`
* [Pow](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Pow)
    * 2nd input must be `2.0`
* [PRelu](https://github.com/onnx/onnx/blob/master/docs/Operators.md#PRelu)
    * `slope` must be `"constant"` and a single value tensor
* [ReduceMean](https://github.com/onnx/onnx/blob/master/docs/Operators.md#reducemean)
* [ReduceSum](https://github.com/onnx/onnx/blob/master/docs/Operators.md#reducesum)
* [Relu](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Relu)
* [Reshape](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Reshape)
* [Resize](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Resize)
    * must be from opset version >= 11
    * `mode` must be `"linear"` or `"nearest"`
    * `nearest_mode` must be `"floor"`
    * `coordinate_transformation_mode` must be either `"pytorch_half_pixel"` or `"align_corners"` for `"linear"` mode
    * `coordinate_transformation_mode` must be either `"asymmetric"` for `"nearest"` mode
* [Sigmoid](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sigmoid)
* [Sin](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sin)
* [Slice](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Slice)
    * must be from opset version >= 10
    * `starts` must be [Constant](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Constant) value or have initializer value
    * `ends` must be [Constant](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Constant) value or have initializer value
    * `axes` must be [Constant](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Constant) value or have initializer value
    * `steps` is not supported
* [Softmax](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Softmax)
* [Split](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Split)
* [Sqrt](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sqrt)
* [Squeeze](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Squeeze)
* [Sub](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sub)
* [Sum](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sum)
    * 2 inputs
* [Tan](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Tan)
* [Tanh](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Tanh)
* [Transpose](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Transpose)
* [Unsqueeze](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Unsqueeze)
* [Where](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Where)
  * `condition` must be [Constant](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Constant) value
  * `condition` must all true or all false
  * the input value not selected must be constant
