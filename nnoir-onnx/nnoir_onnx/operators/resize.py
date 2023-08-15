from typing import Any, Dict, List, Optional, Tuple

import onnx
from nnoir.functions import Function, Resize2D
from numpy.typing import NDArray

from .utils import *


class OpResize(Op):
    def __init__(self, node: onnx.NodeProto, *args: Any):
        super(OpResize, self).__init__(node, *args)

        if self.opset_version < 11:
            raise UnsupportedONNXOperation(self.node, "only opset_version >= 11 is supported")

        self.coordinate_transformation_mode = b"half_pixel"  # unsupported default ONNX coordinate_transformation_mode
        self.mode = b"nearest"
        self.nearest_mode = b"round_prefer_floor"
        for attr in node.attribute:
            if attr.name == "mode":
                if attr.s == b"linear":
                    self.mode = b"linear"
                elif attr.s == b"nearest":
                    self.mode = b"nearest"
                else:
                    raise UnsupportedONNXOperation(
                        self.node,
                        f"{attr.s.decode('utf-8')} mode is not supported for Resize.",
                    )
            if attr.name == "nearest_mode":
                self.nearest_mode = attr.s
            if attr.name == "coordinate_transformation_mode":
                if attr.s == b"pytorch_half_pixel":
                    self.coordinate_transformation_mode = b"align_centers"
                elif attr.s == b"align_corners":
                    self.coordinate_transformation_mode = b"align_corners"
                elif attr.s == b"asymmetric":
                    self.coordinate_transformation_mode = b"asymmetric"
                else:
                    raise UnsupportedONNXOperation(
                        self.node,
                        f"{attr.s.decode('utf-8')} coordinate_transformation_mode " "is not supported for Resize",
                    )
            if attr.name == "exclude_outside":
                if attr.i != 0:
                    raise UnsupportedONNXOperation(self.node, f"exclude_outside={attr.i} is not supported for Resize")

        if self.mode == b"linear":
            self.interpolation_mode = b"linear"
            if self.coordinate_transformation_mode not in [b"align_centers", b"align_corners"]:
                raise UnsupportedONNXOperation(
                    self.node, f"{self.coordinate_transformation_mode.decode('utf-8')} is not supported for Resize linear mode."
                )
        elif self.mode == b"nearest":
            if self.coordinate_transformation_mode not in [b"asymmetric"]:
                raise UnsupportedONNXOperation(
                    self.node,
                    f"{self.coordinate_transformation_mode.decode('utf-8')} is not supported for Resize nearest mode.",
                )
            if self.nearest_mode == b"floor":
                self.interpolation_mode = b"nearest-floor"
            else:
                raise UnsupportedONNXOperation(
                    self.node, f"{self.nearest_mode.decode('utf-8')} is not supported for Resize nearest mode."
                )

    def to_function(self, env: Dict[str, NDArray[Any]], constants: Dict[str, NDArray[Any]]) -> List[Function]:
        x, *_ = self.node.input
        [y] = self.node.output
        return [
            Resize2D(
                [x],
                list(self.node.output),
                size=tuple(env[y].shape[2:]),
                interpolation_mode=self.interpolation_mode,
                coordinate_transformation_mode=self.coordinate_transformation_mode,
            )
        ]
