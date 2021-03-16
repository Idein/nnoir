# test for the conversion of onnx2onnx & onnx2nnoir

all: models/cloud_automl_fixed.nnoir models/custom_vision_fixed.nnoir

%_fixed.onnx: %.onnx
	onnx2onnx $< -o $@

%.nnoir: %.onnx
	onnx2nnoir $< -o $@ --fix_dimension None=1 --graph_name new_graph
