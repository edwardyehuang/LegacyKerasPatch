"""Wrapper for tensorflow.python.ops.ctc_ops - provided by LegacyKerasPatch.

This module re-exports functions from the real TensorFlow implementation
to enable IDE 'Go to Definition' navigation to original source code.
"""

from tensorflow.python.ops.ctc_ops import ctc_beam_search_decoder_v2
from tensorflow.python.ops.ctc_ops import ctc_greedy_decoder
from tensorflow.python.ops.ctc_ops import ctc_loss_v2
