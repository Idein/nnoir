import setuptools
import onnx2nnoirtools

setuptools.setup(
    name=onnx2nnoirtools.NAME,
    version=onnx2nnoirtools.VERSION,
    description=onnx2nnoirtools.DESCRIPTION,
    packages=setuptools.find_packages(),
    python_requires='>=3.5.*',

    author='Idein Inc.',
    author_email='christian@idein.jp',

    project_urls={
        'Source': 'https://github.com/Idein/onnx2nnoirtools'
    },
    install_requires=[
        'numpy',
        'onnx',
        'nnoir-onnx'
    ],
    scripts=['fixonnx4nnoir']
)
