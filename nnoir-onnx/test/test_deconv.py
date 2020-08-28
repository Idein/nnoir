from nose.tools import raises

import numpy as np

import onnx
from onnx.helper import make_node, make_graph, make_model, make_tensor_value_info
from onnx.numpy_helper import from_array
from onnx import TensorProto

from util import Base

info = make_tensor_value_info


def test_deconv_00():
    '''
    opset version >= 11
    '''

    x_shape = (1, 16, 20, 10)
    w_shape = (16, 1, 3, 3)

    out_shape = (1, 33, 41, 21)

    class DeconvTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("ConvTranspose", inputs=["X", "W"], outputs=["Y"], kernel_shape=[3, 3])
            inputs = [info("X", TensorProto.FLOAT, x_shape)]
            outputs = [info("Y", TensorProto.FLOAT, out_shape)]

            W = from_array(np.random.rand(*w_shape).astype(np.float32), "W")
            graph = make_graph([node], "add_graph", inputs, outputs, initializer=[W])
            model = make_model(graph)
            return model

    x = np.random.rand(*x_shape).astype(np.float32)

    outputs = ["Y"]
    DeconvTester({"X": x}, outputs).run()


def test_deconv_01():
    '''
    opset version >= 11
    `kernel_shape` is ommitted. it should be inferred.
    '''

    x_shape = (1, 16, 50, 100)
    w_shape = (16, 1, 3, 3)

    out_shape = (1, 33, 101, 201)

    class DeconvTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("ConvTranspose", inputs=["X", "W"], outputs=["Y"])
            inputs = [info("X", TensorProto.FLOAT, x_shape)]
            outputs = [info("Y", TensorProto.FLOAT, out_shape)]

            W = from_array(np.random.rand(*w_shape).astype(np.float32), "W")
            graph = make_graph([node], "add_graph", inputs, outputs, initializer=[W])
            model = make_model(graph)
            return model

    x = np.random.rand(*x_shape).astype(np.float32)

    outputs = ["Y"]
    DeconvTester({"X": x}, outputs).run()


def test_deconv_02():
    '''
    opset version >= 11
    make_node with `strides` and `pads`.
    '''

    x_shape = (1, 1, 3, 3)
    w_shape = (1, 2, 3, 3)

    out_shape = (1, 2, 7, 3)

    class DeconvTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            node = onnx.helper.make_node("ConvTranspose", ["X", "W"], ["Y"],
                                         strides=[3, 2], pads=[1, 2, 1, 2])
            inputs = [info("X", TensorProto.FLOAT, x_shape)]
            outputs = [info("Y", TensorProto.FLOAT, out_shape)]

            W = from_array(np.random.rand(*w_shape).astype(np.float32), "W")
            graph = make_graph([node], "add_graph", inputs, outputs, initializer=[W])
            model = make_model(graph)
            return model

    x = np.random.rand(*x_shape).astype(np.float32)

    outputs = ["Y"]
    DeconvTester({"X": x}, outputs).run()


@raises(Exception)
def test_deconv_03():
    '''
    opset version >= 11
    make_node with `dialation` and `output_shape`.
    `pads` should be inferred from `output_shape`.
    '''

    x_shape = (1, 1, 3, 3)
    w_shape = (1, 2, 3, 3)

    out_shape = (1, 2, 10, 8)

    class DeconvTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            node = onnx.helper.make_node("ConvTranspose", ["X", "W"], ["Y"],
                                         strides=[3, 2],
                                         output_shape=[10, 8])
            inputs = [info("X", TensorProto.FLOAT, x_shape)]
            outputs = [info("Y", TensorProto.FLOAT, out_shape)]

            W = from_array(np.random.rand(*w_shape).astype(np.float32), "W")
            graph = make_graph([node], "add_graph", inputs, outputs, initializer=[W])
            model = make_model(graph)
            return model

    x = np.random.rand(*x_shape).astype(np.float32)

    outputs = ["Y"]
    DeconvTester({"X": x}, outputs).run()


@raises(Exception)
def test_deconv_04():
    '''
    opset version >= 11
    make_node with `output_padding`.
    '''

    x_shape = (1, 1, 3, 3)
    w_shape = (1, 2, 3, 3)

    out_shape = (1, 2, 2, 2)

    class DeconvTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            node = onnx.helper.make_node("ConvTranspose", ["X", "W"], ["Y"],
                                         output_padding=[1, 1])

            inputs = [info("X", TensorProto.FLOAT, x_shape)]
            outputs = [info("Y", TensorProto.FLOAT, out_shape)]

            W = from_array(np.random.rand(*w_shape).astype(np.float32), "W")
            graph = make_graph([node], "add_graph", inputs, outputs, initializer=[W])
            model = make_model(graph)
            return model

    x = np.random.rand(*x_shape).astype(np.float32)

    outputs = ["Y"]
    DeconvTester({"X": x}, outputs).run()
