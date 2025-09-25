FROM python:3.11-slim-bookworm

ARG NNOIR_VERSION
ARG NNOIR_ONNX_VERSION

COPY ./nnoir/dist/nnoir-${NNOIR_VERSION}-py3-none-any.whl /tmp/nnoir-${NNOIR_VERSION}-py3-none-any.whl
COPY ./nnoir-onnx/dist/nnoir_onnx-${NNOIR_ONNX_VERSION}-py3-none-any.whl /tmp/nnoir_onnx-${NNOIR_ONNX_VERSION}-py3-none-any.whl

RUN pip3 install --break-system-packages /tmp/nnoir-${NNOIR_VERSION}-py3-none-any.whl /tmp/nnoir_onnx-${NNOIR_ONNX_VERSION}-py3-none-any.whl

WORKDIR "/work"
CMD ["/usr/bin/bash"]
