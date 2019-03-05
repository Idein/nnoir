all: mobilenetv2-1.0.nnoir

mobilenetv2-1.0.onnx:
	wget -O $@ 'https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-1.0/mobilenetv2-1.0.onnx'

%.nnoir: %.onnx
	onnx2nnoir -o $@ $<
