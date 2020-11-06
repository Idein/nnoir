all: super_resolution.nnoir

super_resolution.onnx:
	wget -O $@ https://github.com/onnx/models/raw/master/vision/super_resolution/sub_pixel_cnn_2016/model/super-resolution-10.onnx

super_resolution.fixed.onnx: super_resolution.onnx
	freeze_onnx freeze -o $@ --fix-dimension batch_size=1 $<

super_resolution.nnoir: super_resolution.fixed.onnx
	onnx2nnoir -o $@ --graph_name torch_jit_export $<
