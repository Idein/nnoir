import setuptools
import blackonnxlib

setuptools.setup(
    name=blackonnxlib.NAME,
    version=blackonnxlib.VERSION,
    description=blackonnxlib.DESCRIPTION,
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
    scripts=['blackonnx']
)
