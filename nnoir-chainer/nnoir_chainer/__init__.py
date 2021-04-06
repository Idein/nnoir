# Chainer to NNOIR
import nnoir_chainer.configuration
import nnoir_chainer.functions
import nnoir_chainer.links
from nnoir_chainer import _version

from .graph import Graph

# NNOIR to Chainer
from .nnoir_function import NNOIRFunction

__version__ = _version.__version__
