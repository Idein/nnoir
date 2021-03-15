#!/bin/bash

models="models/cloud_automl.onnx models/custom_vision.onnx"

a=0

# echo 'Applying fixes...'
onnx2onnx models/cloud_automl.onnx --fixes fix_quantize fix_name
a=$(($a + $?))
onnx2onnx models/custom_vision.onnx --fixes fix_postprocess fix_name
a=$(($a + $?))

# echo 'Converting to nnoir...'
for model in $models; do
    onnx2nnoir -o ${model::(-5)}.nnoir ${model::(-5)}_fixed.onnx
    a=$(($a + $?))
done

if [ $a -eq 0 ]; then
    echo "Test Passed"
    for model in $models; do
        rm ${model::(-5)}.nnoir ${model::(-5)}_fixed.onnx
    done
fi
