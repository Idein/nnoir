import argparse

import numpy as np
import onnxruntime
from PIL import Image


def inference(options):

    session = onnxruntime.InferenceSession(options.model, None)
    input_name = session.get_inputs()[0].name
    output_names = [el.name for el in session.get_outputs()]

    img = Image.open(options.image)
    img = img.resize(tuple(session.get_inputs()[0].shape[-2:]))
    img_data = np.array(img).astype("float32")
    img_data = img_data.transpose(2, 0, 1)[None, ...]
    if options.bgr:
        img_data = img_data[:, ::-1, :, :]  # RGB to BGR
    if options.normalized:
        img_data = img_data / 255.0

    res = session.run(output_names, {input_name: img_data})
    print("inference outputs: {}".format(res))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inference script using onnxruntime")
    parser.add_argument(
        "--model",
        default="model.onnx",
        help="onnx model file to perform inference with",
    )
    parser.add_argument(
        "--image",
        default="dandelion_sample.jpg",
        help="image file to use as model input",
    )
    parser.add_argument(
        "--bgr",
        action="store_true",
        help="use input image in BGR channel order (default RGB)",
    )
    parser.add_argument(
        "--normalized",
        action="store_true",
        help="scale pixel values in [0, 1] (default [0, 255])",
    )
    args = parser.parse_args()
    inference(args)
