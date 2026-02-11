"""
NumPy-like operations for Keras 2 compatibility.

This module provides keras.ops.numpy compatible operations by wrapping TensorFlow functions.
"""

import tensorflow as tf
import numpy as np

__all__ = [
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
]


def abs(x):
    """Compute element-wise absolute value."""
    return tf.abs(x)


def absolute(x):
    """Compute element-wise absolute value (alias for abs)."""
    return tf.abs(x)


def add(x1, x2):
    """Add two tensors element-wise."""
    return tf.add(x1, x2)


def all(x, axis=None, keepdims=False):
    """Test whether all elements evaluate to True."""
    return tf.reduce_all(tf.cast(x, tf.bool), axis=axis, keepdims=keepdims)


def amax(x, axis=None, keepdims=False):
    """Return the maximum of an array or maximum along an axis."""
    return tf.reduce_max(x, axis=axis, keepdims=keepdims)


def amin(x, axis=None, keepdims=False):
    """Return the minimum of an array or minimum along an axis."""
    return tf.reduce_min(x, axis=axis, keepdims=keepdims)


def angle(x):
    """Return the angle of a complex number."""
    return tf.math.angle(x)


def any(x, axis=None, keepdims=False):
    """Test whether any element evaluates to True."""
    return tf.reduce_any(tf.cast(x, tf.bool), axis=axis, keepdims=keepdims)


def append(x1, x2, axis=None):
    """Append values to the end of an array."""
    if axis is None:
        x1 = tf.reshape(x1, [-1])
        x2 = tf.reshape(x2, [-1])
        return tf.concat([x1, x2], axis=0)
    return tf.concat([x1, x2], axis=axis)


def arange(start, stop=None, step=1, dtype=None):
    """Return evenly spaced values within a given interval."""
    if stop is None:
        start, stop = 0, start
    return tf.range(start, stop, step, dtype=dtype)


def arccos(x):
    """Compute element-wise inverse cosine."""
    return tf.acos(x)


def arccosh(x):
    """Compute element-wise inverse hyperbolic cosine."""
    return tf.acosh(x)


def arcsin(x):
    """Compute element-wise inverse sine."""
    return tf.asin(x)


def arcsinh(x):
    """Compute element-wise inverse hyperbolic sine."""
    return tf.asinh(x)


def arctan(x):
    """Compute element-wise inverse tangent."""
    return tf.atan(x)


def arctan2(y, x):
    """Compute element-wise arc tangent of y/x."""
    return tf.atan2(y, x)


def arctanh(x):
    """Compute element-wise inverse hyperbolic tangent."""
    return tf.atanh(x)


def argmax(x, axis=None, keepdims=False):
    """Return indices of the maximum values along an axis."""
    result = tf.argmax(x, axis=axis)
    if keepdims:
        result = tf.expand_dims(result, axis=axis)
    return result


def argmin(x, axis=None, keepdims=False):
    """Return indices of the minimum values along an axis."""
    result = tf.argmin(x, axis=axis)
    if keepdims:
        result = tf.expand_dims(result, axis=axis)
    return result


def argpartition(x, kth, axis=-1):
    """Return indices that would partition an array."""
    # TensorFlow doesn't have a direct argpartition, use argsort
    return tf.argsort(x, axis=axis)


def argsort(x, axis=-1):
    """Return indices that would sort an array."""
    return tf.argsort(x, axis=axis)


def array(x, dtype=None):
    """Create a tensor."""
    return tf.convert_to_tensor(x, dtype=dtype)


def array_split(x, indices_or_sections, axis=0):
    """Split an array into multiple sub-arrays."""
    if isinstance(indices_or_sections, int):
        return tf.split(x, indices_or_sections, axis=axis)
    else:
        # Handle split at indices
        indices = list(indices_or_sections)
        sizes = []
        prev = 0
        shape = tf.shape(x)[axis]
        for idx in indices:
            sizes.append(idx - prev)
            prev = idx
        sizes.append(shape - prev)
        return tf.split(x, sizes, axis=axis)


def average(x, axis=None, weights=None, keepdims=False):
    """Compute the weighted average along the specified axis."""
    if weights is None:
        return tf.reduce_mean(x, axis=axis, keepdims=keepdims)
    else:
        weighted_sum = tf.reduce_sum(x * weights, axis=axis, keepdims=keepdims)
        sum_weights = tf.reduce_sum(weights, axis=axis, keepdims=keepdims)
        return weighted_sum / sum_weights


def bartlett(M):
    """Return the Bartlett window."""
    return tf.signal.bartlett_window(M)


def bincount(x, weights=None, minlength=0):
    """Count occurrences of each value in array of non-negative ints."""
    return tf.math.bincount(x, weights=weights, minlength=minlength)


def bitwise_and(x1, x2):
    """Compute the bit-wise AND of two arrays element-wise."""
    return tf.bitwise.bitwise_and(x1, x2)


def bitwise_invert(x):
    """Compute bit-wise inversion (alias for bitwise_not)."""
    return tf.bitwise.invert(x)


def bitwise_left_shift(x, n):
    """Shift the bits of an integer to the left."""
    return tf.bitwise.left_shift(x, n)


def bitwise_not(x):
    """Compute bit-wise NOT of an array element-wise."""
    return tf.bitwise.invert(x)


def bitwise_or(x1, x2):
    """Compute the bit-wise OR of two arrays element-wise."""
    return tf.bitwise.bitwise_or(x1, x2)


def bitwise_right_shift(x, n):
    """Shift the bits of an integer to the right."""
    return tf.bitwise.right_shift(x, n)


def bitwise_xor(x1, x2):
    """Compute the bit-wise XOR of two arrays element-wise."""
    return tf.bitwise.bitwise_xor(x1, x2)


def blackman(M):
    """Return the Blackman window."""
    return tf.signal.blackman_window(M)


def broadcast_to(x, shape):
    """Broadcast an array to a new shape."""
    return tf.broadcast_to(x, shape)


def cbrt(x):
    """Return the cube-root of an array, element-wise."""
    return tf.math.pow(tf.cast(x, tf.float32), 1.0 / 3.0)


def ceil(x):
    """Return the ceiling of the input, element-wise."""
    return tf.math.ceil(x)


def clip(x, x_min, x_max):
    """Clip values to a specified range."""
    return tf.clip_by_value(x, x_min, x_max)


def concatenate(arrays, axis=0):
    """Join a sequence of arrays along an existing axis."""
    return tf.concat(arrays, axis=axis)


def conj(x):
    """Return the complex conjugate, element-wise."""
    return tf.math.conj(x)


def conjugate(x):
    """Return the complex conjugate (alias for conj)."""
    return tf.math.conj(x)


def copy(x):
    """Return an array copy of the given object."""
    return tf.identity(x)


def corrcoef(x, rowvar=True):
    """Return correlation coefficients."""
    if not rowvar:
        x = tf.transpose(x)
    
    # Compute covariance matrix
    mean = tf.reduce_mean(x, axis=1, keepdims=True)
    x_centered = x - mean
    cov = tf.matmul(x_centered, x_centered, transpose_b=True) / tf.cast(tf.shape(x)[1] - 1, x.dtype)
    
    # Normalize to get correlation
    std = tf.sqrt(tf.linalg.diag_part(cov))
    std_outer = tf.tensordot(std, std, axes=0)
    return cov / std_outer


def correlate(a, v, mode="valid"):
    """Cross-correlation of two 1-dimensional sequences."""
    a = tf.cast(a, tf.float32)
    v = tf.cast(v, tf.float32)
    
    # Reshape for conv1d: (batch, length, channels)
    a = tf.reshape(a, [1, -1, 1])
    v = tf.reshape(v[::-1], [-1, 1, 1])  # Flip and reshape as kernel
    
    if mode == "valid":
        padding = "VALID"
    elif mode == "same":
        padding = "SAME"
    else:  # full
        # Pad input
        pad_size = tf.shape(v)[0] - 1
        a = tf.pad(a, [[0, 0], [pad_size, pad_size], [0, 0]])
        padding = "VALID"
    
    result = tf.nn.conv1d(a, v, stride=1, padding=padding)
    return tf.reshape(result, [-1])


def cos(x):
    """Cosine element-wise."""
    return tf.cos(x)


def cosh(x):
    """Hyperbolic cosine, element-wise."""
    return tf.cosh(x)


def count_nonzero(x, axis=None, keepdims=False):
    """Count the number of non-zero values in an array."""
    return tf.math.count_nonzero(x, axis=axis, keepdims=keepdims)


def cross(x1, x2, axisa=-1, axisb=-1, axisc=-1, axis=None):
    """Return the cross product of two vectors."""
    return tf.linalg.cross(x1, x2)


def cumprod(x, axis=None, dtype=None):
    """Return the cumulative product of elements along a given axis."""
    if axis is None:
        x = tf.reshape(x, [-1])
        axis = 0
    result = tf.math.cumprod(x, axis=axis)
    if dtype is not None:
        result = tf.cast(result, dtype)
    return result


def cumsum(x, axis=None, dtype=None):
    """Return the cumulative sum of elements along a given axis."""
    if axis is None:
        x = tf.reshape(x, [-1])
        axis = 0
    result = tf.math.cumsum(x, axis=axis)
    if dtype is not None:
        result = tf.cast(result, dtype)
    return result


def deg2rad(x):
    """Convert angles from degrees to radians."""
    return x * (np.pi / 180.0)


def diag(v, k=0):
    """Extract a diagonal or construct a diagonal array."""
    if len(v.shape) == 1:
        return tf.linalg.diag(v, k=k)
    else:
        return tf.linalg.diag_part(v, k=k)


def diagflat(v, k=0):
    """Create a 2-D array with the flattened input as a diagonal."""
    v_flat = tf.reshape(v, [-1])
    return tf.linalg.diag(v_flat, k=k)


def diagonal(a, offset=0, axis1=0, axis2=1):
    """Return specified diagonals."""
    # Transpose to bring axis1 and axis2 to the last two dimensions
    ndim = len(a.shape)
    perm = [i for i in range(ndim) if i not in [axis1, axis2]] + [axis1, axis2]
    a = tf.transpose(a, perm)
    return tf.linalg.diag_part(a, k=offset)


def diff(a, n=1, axis=-1):
    """Calculate the n-th discrete difference along the given axis."""
    for _ in range(n):
        a = tf.experimental.numpy.diff(a, axis=axis)
    return a


def digitize(x, bins, right=False):
    """Return indices of bins to which each value belongs."""
    x = tf.cast(x, tf.float32)
    bins = tf.cast(bins, tf.float32)
    
    if right:
        return tf.searchsorted(bins, x, side='right')
    else:
        return tf.searchsorted(bins, x, side='left')


def divide(x1, x2):
    """Divide arguments element-wise."""
    return tf.divide(x1, x2)


def divide_no_nan(x1, x2):
    """Divide arguments element-wise, returning 0 where denominator is 0."""
    return tf.math.divide_no_nan(x1, x2)


def dot(a, b):
    """Dot product of two arrays."""
    if len(a.shape) == 1 and len(b.shape) == 1:
        return tf.tensordot(a, b, axes=1)
    elif len(a.shape) == 2 and len(b.shape) == 2:
        return tf.matmul(a, b)
    else:
        return tf.tensordot(a, b, axes=[[-1], [-2]])


def einsum(subscripts, *operands):
    """Evaluates the Einstein summation convention."""
    return tf.einsum(subscripts, *operands)


def empty(shape, dtype=None):
    """Return a new array of given shape and type, without initializing entries."""
    dtype = dtype or tf.float32
    return tf.zeros(shape, dtype=dtype)


def empty_like(x, dtype=None):
    """Return a new array with the same shape and type as a given array."""
    dtype = dtype or x.dtype
    return tf.zeros_like(x, dtype=dtype)


def equal(x1, x2):
    """Return (x1 == x2) element-wise."""
    return tf.equal(x1, x2)


def exp(x):
    """Calculate the exponential of all elements in the input array."""
    return tf.exp(x)


def exp2(x):
    """Calculate 2**x for all x in the input array."""
    return tf.math.pow(2.0, tf.cast(x, tf.float32))


def expand_dims(x, axis):
    """Expand the shape of an array."""
    return tf.expand_dims(x, axis)


def expm1(x):
    """Calculate exp(x) - 1 for all elements in the array."""
    return tf.math.expm1(x)


def eye(N, M=None, k=0, dtype=None):
    """Return a 2-D array with ones on the diagonal and zeros elsewhere."""
    dtype = dtype or tf.float32
    if M is None:
        M = N
    return tf.eye(N, M, dtype=dtype)


def flip(x, axis=None):
    """Reverse the order of elements in an array along the given axis."""
    if axis is None:
        return tf.reverse(x, tf.range(len(x.shape)))
    return tf.reverse(x, [axis] if isinstance(axis, int) else list(axis))


def floor(x):
    """Return the floor of the input, element-wise."""
    return tf.floor(x)


def floor_divide(x1, x2):
    """Return the largest integer smaller or equal to the division."""
    return tf.math.floordiv(x1, x2)


def full(shape, fill_value, dtype=None):
    """Return a new array of given shape and type, filled with fill_value."""
    dtype = dtype or tf.float32
    return tf.fill(shape, tf.cast(fill_value, dtype))


def full_like(x, fill_value, dtype=None):
    """Return a full array with the same shape and type as a given array."""
    dtype = dtype or x.dtype
    return tf.fill(tf.shape(x), tf.cast(fill_value, dtype))


def gcd(x1, x2):
    """Return the greatest common divisor of x1 and x2."""
    return tf.math.gcd(x1, x2)


def get_item(x, key):
    """Get item from tensor."""
    return x[key]


def greater(x1, x2):
    """Return the truth value of (x1 > x2) element-wise."""
    return tf.greater(x1, x2)


def greater_equal(x1, x2):
    """Return the truth value of (x1 >= x2) element-wise."""
    return tf.greater_equal(x1, x2)


def hamming(M):
    """Return the Hamming window."""
    return tf.signal.hamming_window(M)


def hanning(M):
    """Return the Hanning window."""
    return tf.signal.hann_window(M)


def heaviside(x1, x2):
    """Compute the Heaviside step function."""
    return tf.where(x1 < 0, tf.zeros_like(x1), tf.where(x1 == 0, x2, tf.ones_like(x1)))


def histogram(x, bins=10, range=None):
    """Compute the histogram of a dataset."""
    return tf.histogram_fixed_width(x, range or [tf.reduce_min(x), tf.reduce_max(x)], nbins=bins)


def hstack(arrays):
    """Stack arrays in sequence horizontally (column wise)."""
    return tf.concat(arrays, axis=1 if len(arrays[0].shape) > 1 else 0)


def hypot(x1, x2):
    """Given the legs of a right triangle, return its hypotenuse."""
    return tf.sqrt(tf.square(x1) + tf.square(x2))


def identity(n, dtype=None):
    """Return the identity array."""
    dtype = dtype or tf.float32
    return tf.eye(n, dtype=dtype)


def imag(x):
    """Return the imaginary part of the complex argument."""
    return tf.math.imag(x)


def inner(a, b):
    """Inner product of two arrays."""
    return tf.tensordot(a, b, axes=[[-1], [-1]])


def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    """Returns a boolean array where two arrays are element-wise equal."""
    result = tf.abs(a - b) <= (atol + rtol * tf.abs(b))
    if equal_nan:
        nan_mask = tf.math.is_nan(a) & tf.math.is_nan(b)
        result = result | nan_mask
    return result


def isfinite(x):
    """Test element-wise for finiteness."""
    return tf.math.is_finite(x)


def isin(element, test_elements, assume_unique=False, invert=False):
    """Test whether elements are in test_elements."""
    element_flat = tf.reshape(element, [-1])
    test_flat = tf.reshape(test_elements, [-1])
    
    result = tf.reduce_any(tf.equal(element_flat[:, tf.newaxis], test_flat), axis=1)
    result = tf.reshape(result, tf.shape(element))
    
    if invert:
        return ~result
    return result


def isinf(x):
    """Test element-wise for positive or negative infinity."""
    return tf.math.is_inf(x)


def isnan(x):
    """Test element-wise for NaN."""
    return tf.math.is_nan(x)


def isneginf(x):
    """Test element-wise for negative infinity."""
    return x == float('-inf')


def isposinf(x):
    """Test element-wise for positive infinity."""
    return x == float('inf')


def isreal(x):
    """Test element-wise if input is real."""
    if x.dtype.is_complex:
        return tf.equal(tf.math.imag(x), 0)
    return tf.ones(tf.shape(x), dtype=tf.bool)


def kaiser(M, beta):
    """Return the Kaiser window."""
    return tf.signal.kaiser_window(M, beta=beta)


def kron(a, b):
    """Kronecker product of two arrays."""
    return tf.linalg.experimental.kron(a, b) if hasattr(tf.linalg.experimental, 'kron') else _kron_fallback(a, b)


def _kron_fallback(a, b):
    """Fallback implementation of Kronecker product."""
    a_shape = tf.shape(a)
    b_shape = tf.shape(b)
    
    # Reshape and tile
    a_expanded = a[:, tf.newaxis, :, tf.newaxis]
    b_expanded = b[tf.newaxis, :, tf.newaxis, :]
    
    result = a_expanded * b_expanded
    result = tf.reshape(result, [a_shape[0] * b_shape[0], a_shape[1] * b_shape[1]])
    return result


def lcm(x1, x2):
    """Return the lowest common multiple of x1 and x2."""
    return tf.abs(x1 * x2) // tf.math.gcd(x1, x2)


def ldexp(x1, x2):
    """Compute x1 * 2**x2."""
    return x1 * tf.math.pow(2.0, tf.cast(x2, tf.float32))


def left_shift(x, n):
    """Shift the bits of an integer to the left."""
    return tf.bitwise.left_shift(x, n)


def less(x1, x2):
    """Return the truth value of (x1 < x2) element-wise."""
    return tf.less(x1, x2)


def less_equal(x1, x2):
    """Return the truth value of (x1 <= x2) element-wise."""
    return tf.less_equal(x1, x2)


def linspace(start, stop, num=50, endpoint=True, dtype=None):
    """Return evenly spaced numbers over a specified interval."""
    return tf.linspace(start, stop, num)


def log(x):
    """Natural logarithm, element-wise."""
    return tf.math.log(x)


def log10(x):
    """Return the base 10 logarithm of the input array."""
    return tf.math.log(x) / tf.math.log(10.0)


def log1p(x):
    """Return the natural logarithm of one plus the input array."""
    return tf.math.log1p(x)


def log2(x):
    """Base-2 logarithm of x."""
    return tf.math.log(x) / tf.math.log(2.0)


def logaddexp(x1, x2):
    """Logarithm of the sum of exponentiations of the inputs."""
    return tf.math.reduce_logsumexp(tf.stack([x1, x2], axis=0), axis=0)


def logaddexp2(x1, x2):
    """Logarithm of the sum of exponentiations of inputs in base-2."""
    return logaddexp(x1 * tf.math.log(2.0), x2 * tf.math.log(2.0)) / tf.math.log(2.0)


def logical_and(x1, x2):
    """Compute the truth value of x1 AND x2 element-wise."""
    return tf.logical_and(x1, x2)


def logical_not(x):
    """Compute the truth value of NOT x element-wise."""
    return tf.logical_not(x)


def logical_or(x1, x2):
    """Compute the truth value of x1 OR x2 element-wise."""
    return tf.logical_or(x1, x2)


def logical_xor(x1, x2):
    """Compute the truth value of x1 XOR x2 element-wise."""
    return tf.logical_xor(x1, x2)


def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None):
    """Return numbers spaced evenly on a log scale."""
    lin = tf.linspace(start, stop, num)
    return tf.math.pow(base, lin)


def matmul(x1, x2):
    """Matrix product of two arrays."""
    return tf.matmul(x1, x2)


def max(x, axis=None, keepdims=False):
    """Return the maximum of an array or maximum along an axis."""
    return tf.reduce_max(x, axis=axis, keepdims=keepdims)


def maximum(x1, x2):
    """Element-wise maximum of array elements."""
    return tf.maximum(x1, x2)


def mean(x, axis=None, keepdims=False):
    """Compute the arithmetic mean along the specified axis."""
    return tf.reduce_mean(x, axis=axis, keepdims=keepdims)


def median(x, axis=None, keepdims=False):
    """Compute the median along the specified axis."""
    if axis is None:
        x = tf.reshape(x, [-1])
        axis = 0
    
    # Sort and get middle value
    sorted_x = tf.sort(x, axis=axis)
    n = tf.shape(x)[axis]
    
    mid = n // 2
    if n % 2 == 0:
        # Average of two middle values
        lower = tf.gather(sorted_x, mid - 1, axis=axis)
        upper = tf.gather(sorted_x, mid, axis=axis)
        result = (lower + upper) / 2
    else:
        result = tf.gather(sorted_x, mid, axis=axis)
    
    if keepdims:
        result = tf.expand_dims(result, axis=axis)
    
    return result


def meshgrid(*arrays, indexing="xy"):
    """Return coordinate matrices from coordinate vectors."""
    return tf.meshgrid(*arrays, indexing=indexing)


def min(x, axis=None, keepdims=False):
    """Return the minimum of an array or minimum along an axis."""
    return tf.reduce_min(x, axis=axis, keepdims=keepdims)


def minimum(x1, x2):
    """Element-wise minimum of array elements."""
    return tf.minimum(x1, x2)


def mod(x1, x2):
    """Return element-wise remainder of division."""
    return tf.math.mod(x1, x2)


def moveaxis(x, source, destination):
    """Move axes of an array to new positions."""
    return tf.experimental.numpy.moveaxis(x, source, destination)


def multiply(x1, x2):
    """Multiply arguments element-wise."""
    return tf.multiply(x1, x2)


def nan_to_num(x, nan=0.0, posinf=None, neginf=None):
    """Replace NaN with zero and infinity with finite numbers."""
    result = tf.where(tf.math.is_nan(x), nan, x)
    if posinf is not None:
        result = tf.where(x == float('inf'), posinf, result)
    if neginf is not None:
        result = tf.where(x == float('-inf'), neginf, result)
    return result


def ndim(x):
    """Return the number of dimensions of an array."""
    return len(x.shape)


def negative(x):
    """Numerical negative, element-wise."""
    return tf.negative(x)


def nonzero(x):
    """Return the indices of the elements that are non-zero."""
    return tf.where(tf.not_equal(x, 0))


def not_equal(x1, x2):
    """Return (x1 != x2) element-wise."""
    return tf.not_equal(x1, x2)


def ones(shape, dtype=None):
    """Return a new array of given shape and type, filled with ones."""
    dtype = dtype or tf.float32
    return tf.ones(shape, dtype=dtype)


def ones_like(x, dtype=None):
    """Return an array of ones with the same shape and type as a given array."""
    dtype = dtype or x.dtype
    return tf.ones_like(x, dtype=dtype)


def outer(a, b):
    """Compute the outer product of two vectors."""
    return tf.tensordot(a, b, axes=0)


def pad(x, pad_width, mode="constant", constant_values=0):
    """Pad an array."""
    return tf.pad(x, pad_width, mode=mode.upper(), constant_values=constant_values)


def power(x1, x2):
    """First array elements raised to powers from second array."""
    return tf.pow(x1, x2)


def prod(x, axis=None, keepdims=False, dtype=None):
    """Return the product of array elements over a given axis."""
    result = tf.reduce_prod(x, axis=axis, keepdims=keepdims)
    if dtype is not None:
        result = tf.cast(result, dtype)
    return result


def quantile(x, q, axis=None, method="linear", keepdims=False):
    """Compute the q-th quantile of the data along the specified axis."""
    if axis is None:
        x = tf.reshape(x, [-1])
        axis = 0
    
    # Sort along axis
    sorted_x = tf.sort(x, axis=axis)
    n = tf.shape(x)[axis]
    
    # Calculate indices
    idx = q * tf.cast(n - 1, tf.float32)
    idx_low = tf.cast(tf.floor(idx), tf.int32)
    idx_high = tf.cast(tf.math.ceil(idx), tf.int32)
    
    low_val = tf.gather(sorted_x, idx_low, axis=axis)
    high_val = tf.gather(sorted_x, idx_high, axis=axis)
    
    # Linear interpolation
    frac = idx - tf.cast(idx_low, tf.float32)
    result = low_val + frac * (high_val - low_val)
    
    if keepdims:
        result = tf.expand_dims(result, axis=axis)
    
    return result


def ravel(x):
    """Return a flattened array."""
    return tf.reshape(x, [-1])


def real(x):
    """Return the real part of the complex argument."""
    return tf.math.real(x)


def reciprocal(x):
    """Return the reciprocal of the argument, element-wise."""
    return tf.math.reciprocal(x)


def repeat(x, repeats, axis=None):
    """Repeat elements of an array."""
    return tf.repeat(x, repeats, axis=axis)


def reshape(x, new_shape):
    """Gives a new shape to an array without changing its data."""
    return tf.reshape(x, new_shape)


def right_shift(x, n):
    """Shift the bits of an integer to the right."""
    return tf.bitwise.right_shift(x, n)


def roll(x, shift, axis=None):
    """Roll array elements along a given axis."""
    return tf.roll(x, shift, axis)


def rot90(x, k=1, axes=(0, 1)):
    """Rotate an array by 90 degrees in the plane specified by axes."""
    return tf.image.rot90(x, k=k)


def round(x, decimals=0):
    """Round an array to the given number of decimals."""
    if decimals == 0:
        return tf.round(x)
    else:
        multiplier = tf.pow(10.0, tf.cast(decimals, tf.float32))
        return tf.round(x * multiplier) / multiplier


def searchsorted(sorted_sequence, values, side="left"):
    """Find indices where elements should be inserted to maintain order."""
    return tf.searchsorted(sorted_sequence, values, side=side)


def select(condlist, choicelist, default=0):
    """Return an array drawn from elements in choicelist, depending on conditions."""
    result = tf.fill(tf.shape(choicelist[0]), tf.cast(default, choicelist[0].dtype))
    for cond, choice in zip(reversed(condlist), reversed(choicelist)):
        result = tf.where(cond, choice, result)
    return result


def sign(x):
    """Returns an element-wise indication of the sign of a number."""
    return tf.sign(x)


def signbit(x):
    """Returns element-wise True where signbit is set (less than zero)."""
    return tf.less(x, 0)


def sin(x):
    """Trigonometric sine, element-wise."""
    return tf.sin(x)


def sinh(x):
    """Hyperbolic sine, element-wise."""
    return tf.sinh(x)


def size(x, axis=None):
    """Return the number of elements along a given axis."""
    if axis is None:
        return tf.size(x)
    return tf.shape(x)[axis]


def slogdet(x):
    """Compute the sign and (natural) logarithm of the determinant."""
    return tf.linalg.slogdet(x)


def sort(x, axis=-1):
    """Return a sorted copy of an array."""
    return tf.sort(x, axis=axis)


def split(x, indices_or_sections, axis=0):
    """Split an array into multiple sub-arrays."""
    return tf.split(x, indices_or_sections, axis=axis)


def sqrt(x):
    """Return the non-negative square-root of an array."""
    return tf.sqrt(x)


def square(x):
    """Return the element-wise square of the input."""
    return tf.square(x)


def squeeze(x, axis=None):
    """Remove single-dimensional entries from the shape of an array."""
    return tf.squeeze(x, axis=axis)


def stack(arrays, axis=0):
    """Join a sequence of arrays along a new axis."""
    return tf.stack(arrays, axis=axis)


def std(x, axis=None, keepdims=False, ddof=0):
    """Compute the standard deviation along the specified axis."""
    mean_val = tf.reduce_mean(x, axis=axis, keepdims=True)
    variance = tf.reduce_mean(tf.square(x - mean_val), axis=axis, keepdims=keepdims)
    
    if ddof != 0:
        n = tf.cast(tf.reduce_prod(tf.gather(tf.shape(x), axis if axis is not None else tf.range(len(x.shape)))), x.dtype)
        variance = variance * n / (n - ddof)
    
    return tf.sqrt(variance)


def subtract(x1, x2):
    """Subtract arguments element-wise."""
    return tf.subtract(x1, x2)


def sum(x, axis=None, keepdims=False, dtype=None):
    """Sum of array elements over a given axis."""
    result = tf.reduce_sum(x, axis=axis, keepdims=keepdims)
    if dtype is not None:
        result = tf.cast(result, dtype)
    return result


def swapaxes(x, axis1, axis2):
    """Interchange two axes of an array."""
    ndim = len(x.shape)
    perm = list(range(ndim))
    perm[axis1], perm[axis2] = perm[axis2], perm[axis1]
    return tf.transpose(x, perm)


def take(x, indices, axis=None):
    """Take elements from an array along an axis."""
    if axis is None:
        return tf.gather(tf.reshape(x, [-1]), indices)
    return tf.gather(x, indices, axis=axis)


def take_along_axis(arr, indices, axis):
    """Take values from the input array by matching 1d index and data slices."""
    return tf.experimental.numpy.take_along_axis(arr, indices, axis)


def tan(x):
    """Compute tangent element-wise."""
    return tf.tan(x)


def tanh(x):
    """Compute hyperbolic tangent element-wise."""
    return tf.tanh(x)


def tensordot(a, b, axes=2):
    """Compute tensor dot product along specified axes."""
    return tf.tensordot(a, b, axes=axes)


def tile(x, reps):
    """Construct an array by repeating x the number of times given by reps."""
    return tf.tile(x, reps)


def trace(x, offset=0, axis1=0, axis2=1):
    """Return the sum along diagonals of the array."""
    return tf.linalg.trace(x)


def transpose(x, axes=None):
    """Permute the dimensions of an array."""
    return tf.transpose(x, perm=axes)


def trapezoid(y, x=None, dx=1.0, axis=-1):
    """Integrate along the given axis using the trapezoidal rule."""
    if x is None:
        d = dx
    else:
        d = tf.experimental.numpy.diff(x, axis=axis)
    
    # Average of consecutive elements
    shape = tf.shape(y)
    ndim = len(y.shape)
    
    # Slice to get y[:-1] and y[1:] along axis
    slices_low = [slice(None)] * ndim
    slices_high = [slice(None)] * ndim
    slices_low[axis] = slice(None, -1)
    slices_high[axis] = slice(1, None)
    
    y_low = y[tuple(slices_low)]
    y_high = y[tuple(slices_high)]
    
    return tf.reduce_sum(d * (y_low + y_high) / 2, axis=axis)


def tri(N, M=None, k=0, dtype=None):
    """An array with ones at and below the given diagonal and zeros elsewhere."""
    dtype = dtype or tf.float32
    if M is None:
        M = N
    return tf.linalg.band_part(tf.ones((N, M), dtype=dtype), -1, k)


def tril(x, k=0):
    """Lower triangle of an array."""
    return tf.linalg.band_part(x, -1, k)


def triu(x, k=0):
    """Upper triangle of an array."""
    return tf.linalg.band_part(x, k, -1)


def true_divide(x1, x2):
    """Returns a true division of the inputs, element-wise."""
    return tf.truediv(x1, x2)


def trunc(x):
    """Return the truncated value of the input, element-wise."""
    return tf.math.floor(tf.where(x >= 0, x, tf.math.ceil(x)))


def unravel_index(indices, shape):
    """Converts a flat index into a tuple of coordinate arrays."""
    return tf.unravel_index(indices, shape)


def vander(x, N=None, increasing=False):
    """Generate a Vandermonde matrix."""
    if N is None:
        N = tf.shape(x)[0]
    
    if increasing:
        powers = tf.range(N)
    else:
        powers = tf.range(N - 1, -1, -1)
    
    x = tf.cast(x, tf.float32)
    return tf.pow(x[:, tf.newaxis], tf.cast(powers, tf.float32))


def var(x, axis=None, keepdims=False, ddof=0):
    """Compute the variance along the specified axis."""
    mean_val = tf.reduce_mean(x, axis=axis, keepdims=True)
    variance = tf.reduce_mean(tf.square(x - mean_val), axis=axis, keepdims=keepdims)
    
    if ddof != 0:
        n = tf.cast(tf.reduce_prod(tf.gather(tf.shape(x), axis if axis is not None else tf.range(len(x.shape)))), x.dtype)
        variance = variance * n / (n - ddof)
    
    return variance


def vdot(a, b):
    """Return the dot product of two vectors."""
    return tf.reduce_sum(tf.reshape(a, [-1]) * tf.reshape(b, [-1]))


def vectorize(pyfunc, otypes=None, signature=None):
    """Generalized function class."""
    return tf.numpy_function(pyfunc, otypes=otypes)


def view(x, dtype):
    """Return a view of the array with the same data but different dtype."""
    return tf.bitcast(x, dtype)


def vstack(arrays):
    """Stack arrays in sequence vertically (row wise)."""
    return tf.concat(arrays, axis=0)


def where(condition, x=None, y=None):
    """Return elements chosen from x or y depending on condition."""
    if x is None and y is None:
        return tf.where(condition)
    return tf.where(condition, x, y)


def zeros(shape, dtype=None):
    """Return a new array of given shape and type, filled with zeros."""
    dtype = dtype or tf.float32
    return tf.zeros(shape, dtype=dtype)


def zeros_like(x, dtype=None):
    """Return an array of zeros with the same shape and type as a given array."""
    dtype = dtype or x.dtype
    return tf.zeros_like(x, dtype=dtype)
