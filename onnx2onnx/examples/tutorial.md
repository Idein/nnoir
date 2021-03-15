# Classification tutorial

The two sample models in `examples/models` were trainined on a simple flower classification task, using data available [here](https://public.roboflow.com/classification/flowers_classification/2). (licence: Public Domain)

## Preparation

You can use the Docker image from `dockerfiles` folder of this repository to go through the examples as it contains all necessary packages.

Then, we need to install the package of this repository, with the following command:

```bash
user$ pip3 install .
```

Azure custom vision allows exporting models directly in onnx format, but google cloud vision does not, so we have to convert the model from tflite format to onnx format using [tflite2onnx](https://github.com/jackwish/tflite2onnx) tool:

```bash
user$ tflite2onnx examples/models/cloud_automl.tflite examples/models/cloud_automl.onnx
```

## Simple inference

Using the script `inference_test.py` we can verify that models work. The image preprocessing differs between models according to the service used to generate it, so use the options consequently:

```bash
user$ cd examples
user$ python3 inference_test.py --model models/cloud_automl.onnx --normalized

> inference outputs: [array([[0.03515625, 0.96484375]], dtype=float32)]

user$ python3 inference_test.py --model models/custom_vision.onnx --bgr

> inference outputs: [array([['Dandelion']], dtype=object), [{'Daisy': 0.24544349312782288, 'Dandelion': 0.7545564770698547}]]
```

## Fix the model

Now, if you try to convert the onnx models to nnoir, it will end with an error code. (you are welcome to try)

The error message may help suggest what fix to apply to the model, and here are the correct fixes for each model:

### Azure custom Vision

`custom_vision.onnx` needs the following fixes:

- correct the graph name
- set static dimensions to the input shape
- use a workaround to avoid unsupported `Split` op
- remove unsupported (and superfluous) post process

This can be done with the following command:

```bash
user$ onnx2onnx models/custom_vision.onnx --fixes fix_postprocess fix_name fix_freeze
```

### Google Cloud Vision

`cloud_automl.onnx` needs the following fixes:

- correct the graph name
- replace unsupported quantization ops

This can be done with the following command:

```bash
user$ onnx2onnx models/cloud_automl.onnx --fixes fix_quantize fix_name
```

Note: `fix_quantize` correction is not strictly equivalent to the original computation, as it requires performing a rounding operation thaht can not be emulated with nnoir supported ops. This affects the precision of the model to a certain extent, but is minor in this task.

## Ready to convert

The fixed models are now ready to be converted!

We can check the models outputs using the same inference script as before, and see that the result is correct (see the note on quantization).

Use `nnoir-onnx` conversion command to get your nnoir model: (this make take some time)

```bash
user$ onnx2nnoir -o models/cloud_automl.nnoir models/cloud_automl_fixed.onnx

user$ onnx2nnoir -o models/custom_vision.nnoir models/custom_vision_fixed.onnx
```

The created models can now be used with [Actcast SDK](https://actcast.io/docs/ForVendor/ApplicationDevelopment/GettingStarted/) to create applications. (available to partners: [partner program](https://actcast.io/docs/files/partner_program.pdf))
