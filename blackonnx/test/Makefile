# test for the conversion of blackonnx & onnx2nnoir

all: models/cloud_automl_fixed.nnoir models/custom_vision_fixed.nnoir

%_fixed.onnx: %.onnx
	blackonnx $< -o $@

%.nnoir: %.onnx
	onnx2nnoir $< -o $@ --fix_dimension None=1 --graph_name new_graph
