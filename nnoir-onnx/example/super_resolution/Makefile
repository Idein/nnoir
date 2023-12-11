all: super_resolution.nnoir

super_resolution.onnx:
	wget -O $@ https://github.com/onnx/models/raw/main/archive/vision/super_resolution/sub_pixel_cnn_2016/model/super-resolution-10.onnx

super_resolution.nnoir: super_resolution.onnx
	onnx2nnoir -o $@ --graph_name torch_jit_export --fix_dimension batch_size=1 $<
