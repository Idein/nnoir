all: custom_vision_fixed.nnoir cloud_automl_fixed.nnoir

custom_vision_fixed.onnx: custom_vision.onnx
custom_vision_fixed.nnoir: custom_vision_fixed.onnx
	onnx2nnoir --fix_dimension "None=1" $< -o $@

cloud_automl.onnx: cloud_automl.tflite
cloud_automl_fixed.onnx: cloud_automl.onnx
cloud_automl_fixed.nnoir: cloud_automl_fixed.onnx


%.onnx: %.tflite
	tflite2onnx $< $@

%_fixed.onnx: %.onnx
	blackonnx $< -o $@
