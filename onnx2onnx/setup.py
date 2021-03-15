import setuptools
import onnx2onnxlib

setuptools.setup(
    name=onnx2onnxlib.NAME,
    version=onnx2onnxlib.VERSION,
    description=onnx2onnxlib.DESCRIPTION,
    packages=setuptools.find_packages(),
    python_requires='>=3.5.*',

    author='Idein Inc.',
    author_email='christian@idein.jp',

    project_urls={
        'Source': 'https://github.com/Idein/onnx2onnxlib'
    },
    install_requires=[
        'numpy',
        'onnx',
        'nnoir-onnx'
    ],
    scripts=['onnx2onnx']
)
