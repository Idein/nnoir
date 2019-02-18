# Chainer to MLIR
import mlir_chainer.configuration
import mlir_chainer.links
import mlir_chainer.functions
from .graph import Graph

# MLIR to Chainer
from .mlir_function import MLIRFunction

from mlir_chainer import _version

__version__ = _version.__version__
