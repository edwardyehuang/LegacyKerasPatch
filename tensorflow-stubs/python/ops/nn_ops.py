"""Wrapper for tensorflow.python.ops.nn_ops - provided by LegacyKerasPatch.

This module re-exports functions from the real TensorFlow implementation
to enable IDE 'Go to Definition' navigation to original source code.
"""

from tensorflow.python.ops.nn_ops import avg_pool_v2
from tensorflow.python.ops.nn_ops import avg_pool1d
from tensorflow.python.ops.nn_ops import avg_pool2d
from tensorflow.python.ops.nn_ops import avg_pool3d
from tensorflow.python.ops.nn_ops import max_pool_v2
from tensorflow.python.ops.nn_ops import max_pool1d
from tensorflow.python.ops.nn_ops import max_pool2d
from tensorflow.python.ops.nn_ops import max_pool3d

from tensorflow.python.ops.nn_ops import conv1d_v2
from tensorflow.python.ops.nn_ops import conv1d_transpose
from tensorflow.python.ops.nn_ops import conv2d_v2
from tensorflow.python.ops.nn_ops import conv2d_transpose_v2
from tensorflow.python.ops.nn_ops import conv3d_v2
from tensorflow.python.ops.nn_ops import conv3d_transpose_v2
from tensorflow.python.ops.nn_ops import depthwise_conv2d_v2
from tensorflow.python.ops.nn_ops import separable_conv2d

from tensorflow.python.ops.nn_ops import bias_add

from tensorflow.python.ops.nn_ops import relu
from tensorflow.python.ops.nn_ops import relu6
from tensorflow.python.ops.nn_ops import leaky_relu
from tensorflow.python.ops.nn_ops import elu
from tensorflow.python.ops.nn_ops import selu
from tensorflow.python.ops.nn_ops import gelu
from tensorflow.python.ops.nn_ops import softmax_v2
from tensorflow.python.ops.nn_ops import log_softmax_v2
from tensorflow.python.ops.nn_ops import softplus
from tensorflow.python.ops.nn_ops import softsign

from tensorflow.python.ops.nn_ops import softmax_cross_entropy_with_logits_v2
from tensorflow.python.ops.nn_ops import sparse_softmax_cross_entropy_with_logits

from tensorflow.python.ops.nn_ops import dropout_v2
