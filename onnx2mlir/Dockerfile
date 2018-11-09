FROM ubuntu:16.04

RUN apt-get update \
 && apt-get upgrade -y \
 && apt-get install -y python3-pip less wget \
 && apt-get clean \
 && apt-get autoclean \
 && apt-get autoremove -y \
 && rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/*

RUN pip3 install onnx \
 && pip3 install onnxruntime \
 && pip3 install cntk \
 && pip3 install msgpack

WORKDIR /root/work
