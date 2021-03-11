FROM ubuntu:bionic
LABEL maintainer="christian@idein.jp"

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update\
 && apt-get install -y python python3 python3-pip curl git
RUN pip3 install --upgrade pip

RUN apt-get install -y libprotobuf-dev protobuf-compiler cmake
RUN pip3 install numpy onnx tflite2onnx nnoir-onnx Pillow onnxruntime
