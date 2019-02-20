# Chainer to NNOIR
import nnoir_chainer.configuration
import nnoir_chainer.links
import nnoir_chainer.functions
from .graph import Graph

# NNOIR to Chainer
from .nnoir_function import NNOIRFunction

from nnoir_chainer import _version

__version__ = _version.__version__
