import os
import sys

import nnoir
import numpy as np


def single_function_model(function, _inputs, _outputs, **kwargs):
    inputs = [nnoir.Value(x[0], shape=x[1], dtype="<f4") for x in _inputs]
    outputs = [nnoir.Value(x[0], shape=x[1], dtype="<f4") for x in _outputs]
    nodes = inputs + outputs
    input_names = [x.name for x in inputs]
    output_names = [x.name for x in outputs]
    functions = [getattr(nnoir.functions, function.partition("_")[0])(input_names, output_names, **kwargs)]
    actual = nnoir.NNOIR(
        function.encode(),
        b"nnoir2chainer_test",
        "0.1",
        input_names,
        output_names,
        nodes,
        functions,
    )
    expected = nnoir.load(os.path.join(os.path.abspath(os.path.dirname(__file__)), function + ".nnoir"))
    xs = [np.random.randn(*x[1]).astype("<f4") for x in _inputs]
    actuals = actual.run(*xs)
    expecteds = expected.run(*xs)
    assert len(expecteds) == len(actuals)
    for a, e in zip(actuals, expecteds):
        print(a - e)
        assert (a == e).all()


def test_Add():
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (10, 10)), (b"v1", (10, 10))],
        [(b"v2", (10, 10))],
    )


def test_AddConstant():
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (10, 10))],
        [(b"v2", (10, 10))],
        value=2.0,
    )


def test_AveragePooling2D():
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (2, 3, 4, 5))],
        [(b"v2", (2, 3, 3, 3))],
        kernel=[2, 3],
        stride=[1, 2],
        pad_h=[0, 0],
        pad_w=[1, 2],
        count_exclude_pad=False,
    )


def test_BatchNormalization():
    shape = (2, 3, 4, 5)
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", shape)],
        [(b"v2", shape)],
        eps=2e-05,
        avg_mean=np.array([6, 7, 8], dtype=np.float32),
        avg_var=np.array([9, 10, 11], dtype=np.float32),
        gamma=np.array([0, 1, 2], dtype=np.float32),
        beta=np.array([3, 4, 5], dtype=np.float32),
    )


def test_Bias():
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (2, 3, 4, 5))],
        [(b"v2", (2, 3, 4, 5))],
        axis=1,
        b=np.arange(3).astype(np.float32),
    )


def test_Bilinear2D():
    in_shape = (2, 3, 9, 10)
    out_shape = (2, 3, 4, 5)
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", in_shape)],
        [(b"v2", out_shape)],
        size=tuple(out_shape[2:]),
    )


def test_Bilinear2D_align_none():
    in_shape = (2, 3, 9, 10)
    out_shape = (2, 3, 4, 5)
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", in_shape)],
        [(b"v2", out_shape)],
        size=tuple(out_shape[2:]),
        mode=b"align_none",
    )


def test_Bilinear2D_align_corners():
    in_shape = (2, 3, 9, 10)
    out_shape = (2, 3, 4, 5)
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", in_shape)],
        [(b"v2", out_shape)],
        size=tuple(out_shape[2:]),
        mode=b"align_corners",
    )


def test_Bilinear2D_align_centers():
    in_shape = (2, 3, 9, 10)
    out_shape = (2, 3, 4, 5)
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", in_shape)],
        [(b"v2", out_shape)],
        size=tuple(out_shape[2:]),
        mode=b"align_centers",
    )


def test_BroadcastTo():
    in_shape = (1, 1, 4, 5)
    out_shape = (2, 3, 4, 5)
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", in_shape)],
        [(b"v2", out_shape)],
        shape=out_shape,
    )


def test_ClippedReLU():
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (10, 10))],
        [(b"v2", (10, 10))],
        upper=40.0,
    )


def test_Concat():
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (10, 11)), (b"v1", (10, 12))],
        [(b"v2", (10, 10))],
        axis=1,
    )


def test_ConstantPadding():
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (2, 3, 4, 5))],
        [(b"v2", (3, 5, 7, 10))],
        pads=((1, 0), (1, 1), (1, 2), (0, 5)),
        value=1.0,
    )


def test_Convolution2D():
    batch = 2
    in_ch = 4
    in_h = 10
    in_w = 9
    out_ch = 7
    out_h = 6
    out_w = 3
    kh = 4
    kw = 3
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (batch, in_ch, in_h, in_w))],
        [(b"v1", (batch, out_ch, out_h, out_w))],
        W=np.arange(out_ch * in_ch * kh * kw).reshape(out_ch, in_ch, kh, kw).astype(np.float32),
        b=np.arange(out_ch).astype(np.float32),
        pad_h=(2, 2),
        pad_w=(1, 1),
        stride=(2, 3),
        dilate=(1, 1),
        groups=1,
    )


def test_Cos():
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (10, 10))],
        [(b"v2", (10, 10))],
    )


def test_DepthwiseConvolution2D():
    sy, sx = 2, 3
    ph, pw = 1, 3
    kh, kw = 3, 4
    batch = 2
    in_ch = 6
    in_h = 10
    in_w = 9
    ch_mul = 2
    out_ch = in_ch * ch_mul
    out_h = 5
    out_w = 4
    dy, dx = (2, 3)
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (batch, in_ch, in_h, in_w))],
        [(b"v1", (batch, out_ch, out_h, out_w))],
        W=np.arange(ch_mul * in_ch * kh * kw).reshape(ch_mul, in_ch, kh, kw).astype(np.float32),
        b=np.arange(out_ch).astype(np.float32),
        pad_h=(1, 1),
        pad_w=(3, 3),
        stride=(sy, sx),
        dilate=(dy, dx),
    )


def test_Div():
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (10, 10)), (b"v1", (10, 10))],
        [(b"v2", (10, 10))],
    )


def test_Dropout():
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (10, 10))],
        [(b"v2", (10, 10))],
    )


def test_ELU():
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (10, 10))],
        [(b"v2", (10, 10))],
        alpha=0.5,
    )


def test_Gemm():
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (3, 4)), (b"v1", (4, 5))],
        [(b"v2", (3, 5))],
        transA=False,
        transB=False,
        alpha=2.0,
        beta=1.0,
    )


def test_LeakyReLU():
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (10, 10))],
        [(b"v2", (10, 10))],
        slope=0.5,
    )


def test_Linear():
    batch = 2
    in_ch = 3
    out_ch = 4
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (batch, in_ch))],
        [(b"v2", (batch, out_ch))],
        W=np.arange(out_ch * in_ch).reshape(out_ch, in_ch).astype(np.float32),
        b=np.arange(out_ch).astype(np.float32),
    )


def test_LocalResponseNormalization():
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (10, 10))],
        [(b"v2", (10, 10))],
        n=6,
        k=3.0,
        alpha=0.0002,
        beta=0.8,
    )


def test_MatMul():
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (3, 4)), (b"v1", (4, 5))],
        [(b"v2", (3, 5))],
    )


def test_MaxPooling2D():
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (2, 3, 4, 5))],
        [(b"v2", (2, 3, 5, 4))],
        kernel=[2, 3],
        stride=[1, 2],
        pad_h=[1, 1],
        pad_w=[2, 3],
    )


def test_Mul():
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (10, 10)), (b"v1", (10, 10))],
        [(b"v2", (10, 10))],
    )


def test_MulConstant():
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (10, 10))],
        [(b"v2", (10, 10))],
        value=2.0,
    )


def test_ReLU():
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (10, 10))],
        [(b"v2", (10, 10))],
    )


def test_Reshape():
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (10, 10))],
        [(b"v2", (5, 20))],
        shape=(5, 20),
    )


def test_Scale():
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (2, 3, 4, 5))],
        [(b"v2", (2, 3, 4, 5))],
        axis=1,
        W=np.arange(3).astype(np.float32),
        b=np.arange(3).astype(np.float32),
    )


def test_Sigmoid():
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (10, 10))],
        [(b"v2", (10, 10))],
    )


def test_Sin():
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (10, 10))],
        [(b"v2", (10, 10))],
    )


def test_Softmax():
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (10, 10))],
        [(b"v2", (10, 10))],
        axis=1,
    )


def test_Sub():
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (10, 10)), (b"v1", (10, 10))],
        [(b"v2", (10, 10))],
    )


def test_Sum():
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (2, 3, 4, 5))],
        [(b"v2", (2, 1, 4, 1))],
        axes=(1, 3),
        keepdims=True,
    )


def test_Swish():
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (10, 10, 3))],
        [(b"v2", (10, 10, 3))],
        beta=np.arange(10).astype(np.float32),
    )


def test_Tan():
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (10, 10))],
        [(b"v2", (10, 10))],
    )


def test_Transpose():
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (2, 3, 4, 5))],
        [(b"v2", (4, 5, 2, 3))],
        axes=(2, 3, 0, 1),
    )


def test_Unpooling2D():
    kh, kw = 2, 3
    sy, sx = 1, 2
    batch = 2
    ch = 3
    in_h, in_w = 5, 6
    out_h, out_w = 4, 9
    ph, pw = 1, 2
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (batch, ch, in_h, in_w))],
        [(b"v2", (batch, ch, out_h, out_w))],
        kh=kh,
        kw=kw,
        sy=sy,
        sx=sx,
        ph=ph,
        pw=pw,
        cover_all=False,
        outh=out_h,
        outw=out_w,
    )
