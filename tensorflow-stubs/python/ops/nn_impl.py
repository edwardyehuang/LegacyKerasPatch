"""Wrapper for tensorflow.python.ops.nn_impl - provided by LegacyKerasPatch.

This module re-exports functions from the real TensorFlow implementation
to enable IDE 'Go to Definition' navigation to original source code.
"""

from tensorflow.python.ops.nn_impl import batch_normalization
from tensorflow.python.ops.nn_impl import l2_normalize_v2
from tensorflow.python.ops.nn_impl import moments_v2

from tensorflow.python.ops.nn_impl import swish
from tensorflow.python.ops.nn_impl import sigmoid
from tensorflow.python.ops.nn_impl import silu
from tensorflow.python.ops.nn_impl import log_sigmoid

from tensorflow.python.ops.nn_impl import sigmoid_cross_entropy_with_logits_v2
