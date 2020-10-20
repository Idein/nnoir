# nnoir-onnx

nnoir-onnx is a converter from ONNX model to NNOIR model.

## Install

```
pip install nnoir-onnx
```

## Example

~~~~bash
wget https://www.cntk.ai/OnnxModels/mnist/opset_7/mnist.tar.gz
tar xvzf mnist.tar.gz
onnx2nnoir -o model.nnoir mnist/model.onnx
~~~~

## Supported ONNX Operators

* [Add](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Add)
* [AveragePool](https://github.com/onnx/onnx/blob/master/docs/Operators.md#AveragePool)
* [BatchNormalization](https://github.com/onnx/onnx/blob/master/docs/Operators.md#BatchNormalization)
    * `scale`, `B`, `mean`, and `var` must be `"constant"`
* [Clip](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Clip)
    * must be opset version 6 or 11
    * if opset version is 11
      * `max` must be `"constant"`
    * `min` must be 0
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
* [Flatten](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Flatten)
* [Gemm](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Gemm)
    * `B` must be [Constant](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Constant) value or have initializer value
    * `C` must be [Constant](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Constant) value or have initializer value
* [GlobalAveragePool](https://github.com/onnx/onnx/blob/master/docs/Operators.md#GlobalAveragePool)
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
    * `dilations` is not supported
* [Mul](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Mul)
* [Pad](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Pad)
    * `mode` must be `"constant"`
* [PRelu](https://github.com/onnx/onnx/blob/master/docs/Operators.md#PRelu)
    * `slope` must be `"constant"` and a single value tensor
* [ReduceMean](https://github.com/onnx/onnx/blob/master/docs/Operators.md#reducemean)
* [ReduceSum](https://github.com/onnx/onnx/blob/master/docs/Operators.md#reducesum)
* [Relu](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Relu)
* [Reshape](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Reshape)
* [Resize](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Resize)
    * must be from opset version >= 11
    * `mode` must be `"linear"`
    * `coordinate_transformation_mode` must be either `"pytorch_half_pixel"` or `"align_corners"`
* [Sigmoid](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sigmoid)
* [Sin](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sin)
* [Softmax](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Softmax)
* [Squeeze](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Squeeze)
* [Sub](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sub)
    * 1st input must not be `"constant"`
* [Sum](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sum)
    * 2 inputs
* [Tan](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Tan)
* [Tanh](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Tanh)
* [Transpose](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Transpose)
* [Unsqueeze](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Unsqueeze)
