"""Wrapper for tensorflow.python.ops.math_ops - provided by LegacyKerasPatch.

This module re-exports functions from the real TensorFlow implementation
to enable IDE 'Go to Definition' navigation to original source code.
"""

from tensorflow.python.ops.math_ops import tanh
