from typing import Any, Dict, List, Optional, Tuple

import onnx
from nnoir.functions import Function
from numpy.typing import NDArray

from .add import OpAdd
from .average_pool import OpAveragePool
from .batch_normalization import OpBatchNormalization
from .clip import OpClip
from .concat import OpConcat
from .conv import OpConv
from .cos import OpCos
from .div import OpDiv
from .dropout import OpDropout
from .elu import OpElu
from .exp import OpExp
from .flatten import OpFlatten
from .gemm import OpGemm
from .global_average_pool import OpGlobalAveragePool
from .hard_sigmoid import OpHardSigmoid
from .hard_swish import OpHardSwish
from .leaky_relu import OpLeakyRelu
from .lrn import OpLRN
from .lstm import OpLSTM
from .mat_mul import OpMatMul
from .max_pool import OpMaxPool
from .mul import OpMul
from .pad import OpPad
from .prelu import OpPRelu
from .reduce_mean import OpReduceMean
from .reduce_sum import OpReduceSum
from .relu import OpRelu
from .reshape import OpReshape
from .resize import OpResize
from .sigmoid import OpSigmoid
from .sin import OpSin
from .softmax import OpSoftmax
from .split import OpSplit
from .squeeze import OpSqueeze
from .sub import OpSub
from .sum import OpSum
from .tan import OpTan
from .tanh import OpTanh
from .transpose import OpTranspose
from .unsqueeze import OpUnsqueeze
