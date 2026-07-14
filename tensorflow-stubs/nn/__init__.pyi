"""Type stubs for tensorflow.nn module - provided by LegacyKerasPatch."""

from typing import Any

# Pooling operations
from tensorflow.python.ops.nn_ops import avg_pool_v2 as avg_pool
from tensorflow.python.ops.nn_ops import avg_pool1d as avg_pool1d
from tensorflow.python.ops.nn_ops import avg_pool2d as avg_pool2d
from tensorflow.python.ops.nn_ops import avg_pool3d as avg_pool3d
from tensorflow.python.ops.nn_ops import max_pool_v2 as max_pool
from tensorflow.python.ops.nn_ops import max_pool1d as max_pool1d
from tensorflow.python.ops.nn_ops import max_pool2d as max_pool2d
from tensorflow.python.ops.nn_ops import max_pool3d as max_pool3d

# Convolution operations
from tensorflow.python.ops.nn_ops import conv1d_v2 as conv1d
from tensorflow.python.ops.nn_ops import conv1d_transpose as conv1d_transpose
from tensorflow.python.ops.nn_ops import conv2d_v2 as conv2d
from tensorflow.python.ops.nn_ops import conv2d_transpose_v2 as conv2d_transpose
from tensorflow.python.ops.nn_ops import conv3d_v2 as conv3d
from tensorflow.python.ops.nn_ops import conv3d_transpose_v2 as conv3d_transpose
from tensorflow.python.ops.nn_ops import depthwise_conv2d_v2 as depthwise_conv2d
from tensorflow.python.ops.nn_ops import separable_conv2d as separable_conv2d

# Normalization and moments
from tensorflow.python.ops.nn_impl import batch_normalization as batch_normalization
from tensorflow.python.ops.nn_impl import l2_normalize_v2 as l2_normalize
from tensorflow.python.ops.nn_impl import moments_v2 as moments

# Bias
from tensorflow.python.ops.nn_ops import bias_add as bias_add

# Activations
from tensorflow.python.ops.nn_ops import relu as relu
from tensorflow.python.ops.nn_ops import relu6 as relu6
from tensorflow.python.ops.nn_ops import leaky_relu as leaky_relu
from tensorflow.python.ops.nn_ops import elu as elu
from tensorflow.python.ops.nn_ops import selu as selu
from tensorflow.python.ops.nn_ops import gelu as gelu
from tensorflow.python.ops.nn_ops import softmax_v2 as softmax
from tensorflow.python.ops.nn_ops import log_softmax_v2 as log_softmax
from tensorflow.python.ops.nn_ops import softplus as softplus
from tensorflow.python.ops.nn_ops import softsign as softsign
from tensorflow.python.ops.nn_impl import swish as swish
from tensorflow.python.ops.nn_impl import sigmoid as sigmoid
from tensorflow.python.ops.math_ops import tanh as tanh
from tensorflow.python.ops.nn_impl import silu as silu
from tensorflow.python.ops.nn_impl import log_sigmoid as log_sigmoid

# Loss functions
from tensorflow.python.ops.nn_impl import sigmoid_cross_entropy_with_logits_v2 as sigmoid_cross_entropy_with_logits
from tensorflow.python.ops.nn_ops import softmax_cross_entropy_with_logits_v2 as softmax_cross_entropy_with_logits
from tensorflow.python.ops.nn_ops import sparse_softmax_cross_entropy_with_logits as sparse_softmax_cross_entropy_with_logits

# CTC operations
from tensorflow.python.ops.ctc_ops import ctc_beam_search_decoder_v2 as ctc_beam_search_decoder
from tensorflow.python.ops.ctc_ops import ctc_greedy_decoder as ctc_greedy_decoder
from tensorflow.python.ops.ctc_ops import ctc_loss_v2 as ctc_loss

# Dropout
from tensorflow.python.ops.nn_ops import dropout_v2 as dropout

def __getattr__(name: str) -> Any: ...
