"""
Keras 3 ops compatibility layer for Keras 2.

This module provides keras.ops compatible operations by wrapping TensorFlow functions,
enabling Keras 2 users to migrate their code to use keras.ops in preparation for Keras 3.
"""

import tensorflow as _tf

# Import submodules
from . import nn
from . import image
from . import linalg
from . import numpy_ops as numpy

# Re-export all numpy operations at the top level (as Keras 3 does)
from .numpy_ops import (
    abs, absolute, add, all, amax, amin, angle, any, append, arange,
    arccos, arccosh, arcsin, arcsinh, arctan, arctan2, arctanh,
    argmax, argmin, argpartition, argsort, array, array_split, average,
    bartlett, bincount,
    bitwise_and, bitwise_invert, bitwise_left_shift, bitwise_not, bitwise_or, bitwise_right_shift, bitwise_xor,
    blackman, broadcast_to, cbrt, ceil, clip, concatenate, conj, conjugate, copy,
    corrcoef, correlate, cos, cosh, count_nonzero, cross, cumprod, cumsum,
    deg2rad, diag, diagflat, diagonal, diff, digitize, divide, divide_no_nan, dot, einsum,
    empty, empty_like, equal, exp, exp2, expand_dims, expm1, eye, flip, floor, floor_divide,
    full, full_like, gcd, get_item, greater, greater_equal,
    hamming, hanning, heaviside, histogram, hstack, hypot, identity, imag, inner,
    isclose, isfinite, isin, isinf, isnan, isneginf, isposinf, isreal, kaiser, kron,
    lcm, ldexp, left_shift, less, less_equal, linspace, log, log10, log1p, log2,
    logaddexp, logaddexp2, logical_and, logical_not, logical_or, logical_xor, logspace,
    matmul, max, maximum, mean, median, meshgrid, min, minimum, mod, moveaxis, multiply,
    nan_to_num, ndim, negative, nonzero, not_equal, ones, ones_like, outer, pad, power, prod,
    quantile, ravel, real, reciprocal, repeat, reshape, right_shift, roll, rot90, round,
    searchsorted, select, sign, signbit, sin, sinh, size, slogdet, sort, split, sqrt, square,
    squeeze, stack, std, subtract, sum, swapaxes, take, take_along_axis, tan, tanh,
    tensordot, tile, trace, transpose, trapezoid, tri, tril, triu, true_divide, trunc,
    unravel_index, vander, var, vdot, vectorize, view, vstack, where, zeros, zeros_like
)

# Re-export nn operations at top level
from .nn import (
    adaptive_average_pool, adaptive_max_pool, average_pool, batch_normalization,
    binary_crossentropy, categorical_crossentropy, celu, conv, conv_transpose,
    ctc_decode, ctc_loss, depthwise_conv, dot_product_attention, elu, gelu, glu,
    hard_shrink, hard_sigmoid, hard_silu, hard_swish, hard_tanh, layer_normalization,
    leaky_relu, log_sigmoid, log_softmax, max_pool, moments, multi_hot, normalize,
    one_hot, polar, psnr, relu, relu6, rms_normalization, selu, separable_conv,
    sigmoid, silu, soft_shrink, softmax, softplus, softsign, sparse_categorical_crossentropy,
    sparse_plus, sparse_sigmoid, sparsemax, squareplus, swish, tanh_shrink, threshold, unfold
)

# Re-export linalg operations
from .linalg import (
    cholesky, cholesky_inverse, det, eig, eigh, inv,
    lstsq, lu_factor, qr, solve, solve_triangular, svd
)
# Note: jvp and norm are handled separately to avoid conflicts

__all__ = [
    # Submodules
    "nn", "image", "linalg", "numpy",
    # NumPy operations re-exported at top level
    "abs", "absolute", "add", "all", "amax", "amin", "angle", "any", "append", "arange",
    "arccos", "arccosh", "arcsin", "arcsinh", "arctan", "arctan2", "arctanh",
    "argmax", "argmin", "argpartition", "argsort", "array", "array_split", "average",
    "bartlett", "bincount",
    "bitwise_and", "bitwise_invert", "bitwise_left_shift", "bitwise_not", "bitwise_or", "bitwise_right_shift", "bitwise_xor",
    "blackman", "broadcast_to", "cbrt", "ceil", "clip", "concatenate", "conj", "conjugate", "copy",
    "corrcoef", "correlate", "cos", "cosh", "count_nonzero", "cross", "cumprod", "cumsum",
    "deg2rad", "diag", "diagflat", "diagonal", "diff", "digitize", "divide", "divide_no_nan", "dot", "einsum",
    "empty", "empty_like", "equal", "exp", "exp2", "expand_dims", "expm1", "eye", "flip", "floor", "floor_divide",
    "full", "full_like", "gcd", "get_item", "greater", "greater_equal",
    "hamming", "hanning", "heaviside", "histogram", "hstack", "hypot", "identity", "imag", "inner",
    "isclose", "isfinite", "isin", "isinf", "isnan", "isneginf", "isposinf", "isreal", "kaiser", "kron",
    "lcm", "ldexp", "left_shift", "less", "less_equal", "linspace", "log", "log10", "log1p", "log2",
    "logaddexp", "logaddexp2", "logical_and", "logical_not", "logical_or", "logical_xor", "logspace",
    "matmul", "max", "maximum", "mean", "median", "meshgrid", "min", "minimum", "mod", "moveaxis", "multiply",
    "nan_to_num", "ndim", "negative", "nonzero", "not_equal", "ones", "ones_like", "outer", "pad", "power", "prod",
    "quantile", "ravel", "real", "reciprocal", "repeat", "reshape", "right_shift", "roll", "rot90", "round",
    "searchsorted", "select", "sign", "signbit", "sin", "sinh", "size", "slogdet", "sort", "split", "sqrt", "square",
    "squeeze", "stack", "std", "subtract", "sum", "swapaxes", "take", "take_along_axis", "tan", "tanh",
    "tensordot", "tile", "trace", "transpose", "trapezoid", "tri", "tril", "triu", "true_divide", "trunc",
    "unravel_index", "vander", "var", "vdot", "vectorize", "view", "vstack", "where", "zeros", "zeros_like",
    # NN operations re-exported at top level
    "adaptive_average_pool", "adaptive_max_pool", "average_pool", "batch_normalization",
    "binary_crossentropy", "categorical_crossentropy", "celu", "conv", "conv_transpose",
    "ctc_decode", "ctc_loss", "depthwise_conv", "dot_product_attention", "elu", "gelu", "glu",
    "hard_shrink", "hard_sigmoid", "hard_silu", "hard_swish", "hard_tanh", "layer_normalization",
    "leaky_relu", "log_sigmoid", "log_softmax", "max_pool", "moments", "multi_hot", "normalize",
    "one_hot", "polar", "psnr", "relu", "relu6", "rms_normalization", "selu", "separable_conv",
    "sigmoid", "silu", "soft_shrink", "softmax", "softplus", "softsign", "sparse_categorical_crossentropy",
    "sparse_plus", "sparse_sigmoid", "sparsemax", "squareplus", "swish", "tanh_shrink", "threshold", "unfold",
    # Linalg operations re-exported at top level
    "cholesky", "cholesky_inverse", "det", "eig", "eigh", "inv",
    "lstsq", "lu_factor", "qr", "solve", "solve_triangular", "svd",
    # Core operations
    "associative_scan", "cast", "cond", "convert_to_numpy", "convert_to_tensor", "custom_gradient",
    "dtype", "erf", "erfinv", "extract_sequences", "fft", "fft2", "fori_loop",
    "ifft2", "in_top_k", "irfft", "is_tensor", "istft", "jvp", "logdet", "logsumexp",
    "map", "norm", "rearrange", "rfft", "rsqrt", "saturate_cast", "scan",
    "scatter", "scatter_update", "segment_max", "segment_sum", "shape", "slice", "slice_update",
    "stft", "stop_gradient", "switch", "top_k", "unstack", "vectorized_map",
    "view_as_complex", "view_as_real", "while_loop",
]


# Additional core operations


def cast(x, dtype):
    """Cast a tensor to a new dtype."""
    return _tf.cast(x, dtype)


def cond(pred, true_fn, false_fn):
    """Conditionally apply true_fn or false_fn."""
    return _tf.cond(pred, true_fn, false_fn)


def convert_to_numpy(x):
    """Convert a tensor to a NumPy array."""
    return x.numpy()


def convert_to_tensor(x, dtype=None):
    """Convert a value to a tensor."""
    return _tf.convert_to_tensor(x, dtype=dtype)


def custom_gradient(f):
    """Decorator to define a function with a custom gradient."""
    return _tf.custom_gradient(f)


def dtype(x):
    """Return the dtype of a tensor."""
    return x.dtype


def erf(x):
    """Compute the error function."""
    return _tf.math.erf(x)


def erfinv(x):
    """Compute the inverse error function."""
    return _tf.math.erfinv(x)


def extract_sequences(x, sequence_length, sequence_stride=1):
    """Extract sequences from a tensor."""
    return _tf.signal.frame(x, sequence_length, sequence_stride)


def fft(x):
    """Compute the Fast Fourier Transform."""
    return _tf.signal.fft(_tf.cast(x, _tf.complex64))


def fft2(x):
    """Compute the 2D Fast Fourier Transform."""
    return _tf.signal.fft2d(_tf.cast(x, _tf.complex64))


def fori_loop(lower, upper, body_fn, init_val):
    """For loop with a body function."""
    return _tf.while_loop(
        lambda i, _: i < upper,
        lambda i, val: (i + 1, body_fn(i, val)),
        [lower, init_val]
    )[1]


def ifft2(x):
    """Compute the 2D Inverse Fast Fourier Transform."""
    return _tf.signal.ifft2d(x)


def in_top_k(targets, predictions, k):
    """Check if targets are in top k predictions."""
    return _tf.math.in_top_k(targets, predictions, k)


def irfft(x, fft_length=None):
    """Compute the Inverse Real Fast Fourier Transform."""
    return _tf.signal.irfft(x, fft_length=fft_length)


def is_tensor(x):
    """Check if x is a tensor."""
    return _tf.is_tensor(x)


def istft(x, sequence_length, sequence_stride, fft_length=None, window="hann", center=True):
    """Compute the Inverse Short-Time Fourier Transform."""
    return _tf.signal.inverse_stft(x, sequence_length, sequence_stride, fft_length)


def jvp(primals, tangents, fn):
    """Compute Jacobian-vector products."""
    return linalg.jvp(primals, tangents, fn)


def logdet(x):
    """Compute the log determinant of a matrix."""
    return _tf.linalg.logdet(x)


def logsumexp(x, axis=None, keepdims=False):
    """Compute the log of the sum of exponentials."""
    return _tf.reduce_logsumexp(x, axis=axis, keepdims=keepdims)


def map(f, xs):
    """Map a function over leading array dimension."""
    return _tf.map_fn(f, xs)


def norm(x, ord=None, axis=None, keepdims=False):
    """Compute the matrix or vector norm."""
    return linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)


def rearrange(x, pattern, **axes_lengths):
    """Rearrange a tensor according to a pattern (simplified implementation)."""
    # This is a simplified version - full implementation would need einops-like parsing
    raise NotImplementedError("rearrange requires einops-like pattern parsing. Consider using _tf.einsum or _tf.transpose directly.")


def rfft(x, fft_length=None):
    """Compute the Real Fast Fourier Transform."""
    return _tf.signal.rfft(x, fft_length=fft_length)


def rsqrt(x):
    """Compute the reciprocal of the square root."""
    return _tf.math.rsqrt(x)


def saturate_cast(x, dtype):
    """Cast with saturation for out-of-range values."""
    return _tf.saturate_cast(x, dtype)


def scan(f, init, xs, length=None, reverse=False, unroll=1):
    """Scan a function over leading array axes."""
    return _tf.scan(f, xs, initializer=init, reverse=reverse)


def scatter(indices, values, shape):
    """Scatter values into a tensor at specified indices."""
    return _tf.scatter_nd(indices, values, shape)


def scatter_update(inputs, indices, updates):
    """Update tensor at specified indices."""
    return _tf.tensor_scatter_nd_update(inputs, indices, updates)


def segment_max(data, segment_ids, num_segments=None, sorted=False):
    """Compute the maximum along segments."""
    if sorted:
        return _tf.math.segment_max(data, segment_ids)
    return _tf.math.unsorted_segment_max(data, segment_ids, num_segments)


def segment_sum(data, segment_ids, num_segments=None, sorted=False):
    """Compute the sum along segments."""
    if sorted:
        return _tf.math.segment_sum(data, segment_ids)
    return _tf.math.unsorted_segment_sum(data, segment_ids, num_segments)


def shape(x):
    """Return the shape of a tensor."""
    return _tf.shape(x)


def slice(x, start, shape):
    """Extract a slice from a tensor."""
    return _tf.slice(x, start, shape)


def slice_update(x, start_indices, values):
    """Update an input tensor starting at the provided indices."""
    rank = x.shape.rank or values.shape.rank
    if rank is None:
        raise ValueError("slice_update requires `x` or `values` to have a known rank.")

    start_indices = _tf.cast(_tf.reshape(_tf.convert_to_tensor(start_indices), [-1]), _tf.int32)
    update_shape = _tf.shape(values, out_type=start_indices.dtype)
    indices = [
        _tf.range(start_indices[i], start_indices[i] + update_shape[i], dtype=start_indices.dtype)
        for i in range(rank)
    ]

    mesh = _tf.meshgrid(*indices, indexing="ij")
    scatter_indices = _tf.stack([_tf.reshape(m, [-1]) for m in mesh], axis=-1)
    return _tf.tensor_scatter_nd_update(x, scatter_indices, _tf.reshape(values, [-1]))


def stft(x, sequence_length, sequence_stride, fft_length=None, window="hann", center=True):
    """Compute the Short-Time Fourier Transform."""
    return _tf.signal.stft(x, sequence_length, sequence_stride, fft_length)


def stop_gradient(x):
    """Stop gradient computation."""
    return _tf.stop_gradient(x)


def switch(index, branches, *operands):
    """Select a branch based on index."""
    return _tf.switch_case(index, {i: lambda b=b: b(*operands) for i, b in enumerate(branches)})


def top_k(x, k, sorted=True):
    """Find top k values and indices."""
    return _tf.math.top_k(x, k=k, sorted=sorted)


def unstack(x, num=None, axis=0):
    """Unpack a tensor along an axis."""
    return _tf.unstack(x, num=num, axis=axis)


def vectorized_map(fn, elems):
    """Vectorized map operation."""
    return _tf.vectorized_map(fn, elems)


def view_as_complex(x):
    """View a real tensor as complex."""
    # Last dimension should be 2 (real, imag)
    return _tf.complex(x[..., 0], x[..., 1])


def view_as_real(x):
    """View a complex tensor as real."""
    return _tf.stack([_tf.math.real(x), _tf.math.imag(x)], axis=-1)


def while_loop(cond, body, loop_vars, maximum_iterations=None):
    """While loop operation."""
    return _tf.while_loop(cond, body, loop_vars, maximum_iterations=maximum_iterations)


def associative_scan(f, elems, axis=0, reverse=False):
    """Perform a parallel associative scan."""
    return _tf.scan(f, elems, reverse=reverse)
