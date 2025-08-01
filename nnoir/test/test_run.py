import os
import sys
from typing import Any, Iterable, Tuple

import nnoir
import numpy as np
import pytest


def single_function_model(
    function: str,
    _inputs: Iterable[Tuple[bytes, Tuple[int, ...]]],
    _outputs: Iterable[Tuple[bytes, Tuple[int, ...]]],
    **kwargs: Any,
) -> None:
    print(f"function: {function}, _inputs: {type(_inputs)},output {type(_outputs)} ")
    inputs = [nnoir.Value(x[0], shape=x[1], dtype=b"<f4") for x in _inputs]
    outputs = [nnoir.Value(x[0], shape=x[1], dtype=b"<f4") for x in _outputs]
    nodes = inputs + outputs
    input_names = [x.name for x in inputs]
    output_names = [x.name for x in outputs]
    functions = [getattr(nnoir.functions, function.partition("_")[0])(input_names, output_names, **kwargs)]
    actual = nnoir.NNOIR(
        function.encode(),
        b"nnoir2chainer_test",
        b"0.1",
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
        if np.isnan(e).any():
            assert (np.isnan(a) == np.isnan(e)).all() and (np.nan_to_num(a) == np.nan_to_num(e)).all()
        else:
            print(a - e)
            assert (a == e).all()


def test_Add() -> None:
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (10, 10)), (b"v1", (10, 10))],
        [(b"v2", (10, 10))],
    )


def test_AddConstant() -> None:
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (10, 10))],
        [(b"v2", (10, 10))],
        value=2.0,
    )


def test_AveragePooling2D() -> None:
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


def test_BatchNormalization() -> None:
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


def test_Bias() -> None:
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (2, 3, 4, 5))],
        [(b"v2", (2, 3, 4, 5))],
        axis=1,
        b=np.arange(3).astype(np.float32),
    )


def test_Bilinear2D() -> None:
    in_shape = (2, 3, 9, 10)
    out_shape = (2, 3, 4, 5)
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", in_shape)],
        [(b"v2", out_shape)],
        size=tuple(out_shape[2:]),
    )


def test_Bilinear2D_align_none() -> None:
    in_shape = (2, 3, 9, 10)
    out_shape = (2, 3, 4, 5)
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", in_shape)],
        [(b"v2", out_shape)],
        size=tuple(out_shape[2:]),
        mode=b"align_none",
    )


def test_Bilinear2D_align_corners() -> None:
    in_shape = (2, 3, 9, 10)
    out_shape = (2, 3, 4, 5)
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", in_shape)],
        [(b"v2", out_shape)],
        size=tuple(out_shape[2:]),
        mode=b"align_corners",
    )


def test_Bilinear2D_align_centers() -> None:
    in_shape = (2, 3, 9, 10)
    out_shape = (2, 3, 4, 5)
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", in_shape)],
        [(b"v2", out_shape)],
        size=tuple(out_shape[2:]),
        mode=b"align_centers",
    )


def test_BroadcastTo() -> None:
    in_shape = (1, 1, 4, 5)
    out_shape = (2, 3, 4, 5)
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", in_shape)],
        [(b"v2", out_shape)],
        shape=out_shape,
    )


def test_ClippedReLU() -> None:
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (10, 10))],
        [(b"v2", (10, 10))],
        upper=40.0,
    )


def test_Concat() -> None:
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (10, 11)), (b"v1", (10, 12))],
        [(b"v2", (10, 10))],
        axis=1,
    )


def test_ConstantPadding() -> None:
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (2, 3, 4, 5))],
        [(b"v2", (3, 5, 7, 10))],
        pads=((1, 0), (1, 1), (1, 2), (0, 5)),
        value=1.0,
    )


def test_Convolution2D() -> None:
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


def test_Cos() -> None:
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (10, 10))],
        [(b"v2", (10, 10))],
    )


def test_DepthwiseConvolution2D() -> None:
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


def test_Div() -> None:
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (10, 10)), (b"v1", (10, 10))],
        [(b"v2", (10, 10))],
    )


def test_Dropout() -> None:
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (10, 10))],
        [(b"v2", (10, 10))],
    )


def test_ELU() -> None:
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (10, 10))],
        [(b"v2", (10, 10))],
        alpha=0.5,
    )


def test_Erf() -> None:
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v1", (1, 3, 4, 5))],
        [(b"v0", (1, 3, 4, 5))],
    )


def test_Exp() -> None:
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v1", (1, 3, 4, 5))],
        [(b"v0", (1, 3, 4, 5))],
    )


def test_Gemm() -> None:
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (3, 4)), (b"v1", (4, 5))],
        [(b"v2", (3, 5))],
        transA=False,
        transB=False,
        alpha=2.0,
        beta=1.0,
    )


def test_LeakyReLU() -> None:
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (10, 10))],
        [(b"v2", (10, 10))],
        slope=0.5,
    )


def test_Linear() -> None:
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


def test_LocalResponseNormalization() -> None:
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (10, 10))],
        [(b"v2", (10, 10))],
        n=6,
        k=3.0,
        alpha=0.0002,
        beta=0.8,
    )


def test_MatMul() -> None:
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (3, 4)), (b"v1", (4, 5))],
        [(b"v2", (3, 5))],
    )


def test_MaxPooling2D() -> None:
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (2, 3, 4, 5))],
        [(b"v2", (2, 3, 5, 4))],
        kernel=[2, 3],
        stride=[1, 2],
        pad_h=[1, 1],
        pad_w=[2, 3],
    )


def test_Mul() -> None:
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (10, 10)), (b"v1", (10, 10))],
        [(b"v2", (10, 10))],
    )


def test_MulConstant() -> None:
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (10, 10))],
        [(b"v2", (10, 10))],
        value=2.0,
    )


def test_ReLU() -> None:
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (10, 10))],
        [(b"v2", (10, 10))],
    )


def test_Reshape() -> None:
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (10, 10))],
        [(b"v2", (5, 20))],
        shape=(5, 20),
    )


def test_Resize2D() -> None:
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (1, 2, 3, 4))],
        [(b"v1", (1, 2, 6, 8))],
        size=[6, 8],
        interpolation_mode=b"nearest-floor",
        coordinate_transformation_mode=b"asymmetric",
    )


def test_Resize2D_linear() -> None:
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (1, 2, 3, 4))],
        [(b"v1", (1, 2, 6, 8))],
        size=[6, 8],
        interpolation_mode=b"linear",
        coordinate_transformation_mode=b"align_centers",
    )


def test_Scale() -> None:
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (2, 3, 4, 5))],
        [(b"v2", (2, 3, 4, 5))],
        axis=1,
        W=np.arange(3).astype(np.float32),
        b=np.arange(3).astype(np.float32),
    )


def test_Sigmoid() -> None:
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (10, 10))],
        [(b"v2", (10, 10))],
    )


def test_Sin() -> None:
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (10, 10))],
        [(b"v2", (10, 10))],
    )


def test_Slice() -> None:
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (2, 4))],
        [(b"v2", (1, 3))],
        starts=[1, 0],
        ends=[2, 3],
    )


def test_Softmax() -> None:
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (10, 10))],
        [(b"v2", (10, 10))],
        axis=1,
    )


@pytest.mark.filterwarnings("ignore: invalid value encountered in sqrt")
def test_Sqrt() -> None:
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (3, 3))],
        [(b"v2", (3, 3))],
    )


def test_Sub() -> None:
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (10, 10)), (b"v1", (10, 10))],
        [(b"v2", (10, 10))],
    )


def test_Sum() -> None:
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (2, 3, 4, 5))],
        [(b"v2", (2, 1, 4, 1))],
        axes=(1, 3),
        keepdims=True,
    )


def test_Swish() -> None:
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (10, 10, 3))],
        [(b"v2", (10, 10, 3))],
        beta=np.arange(10).astype(np.float32),
    )


def test_Tan() -> None:
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (10, 10))],
        [(b"v2", (10, 10))],
    )


def test_Transpose() -> None:
    single_function_model(
        sys._getframe().f_code.co_name[5:],
        [(b"v0", (2, 3, 4, 5))],
        [(b"v2", (4, 5, 2, 3))],
        axes=(2, 3, 0, 1),
    )


def test_Unpooling2D() -> None:
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
