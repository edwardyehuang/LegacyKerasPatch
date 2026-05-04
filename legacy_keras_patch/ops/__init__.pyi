"""Type stubs for legacy_keras_patch.ops."""

from typing import Any, Callable, Optional, Sequence, Tuple, Union

from . import nn as nn
from . import image as image
from . import linalg as linalg
from . import numpy_ops as numpy

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

# Re-exported numpy operations at top level
from .numpy_ops import (
    abs as abs,
    absolute as absolute,
    add as add,
    all as all,
    amax as amax,
    amin as amin,
    angle as angle,
    any as any,
    append as append,
    arange as arange,
    arccos as arccos,
    arccosh as arccosh,
    arcsin as arcsin,
    arcsinh as arcsinh,
    arctan as arctan,
    arctan2 as arctan2,
    arctanh as arctanh,
    argmax as argmax,
    argmin as argmin,
    argpartition as argpartition,
    argsort as argsort,
    array as array,
    array_split as array_split,
    average as average,
    bartlett as bartlett,
    bincount as bincount,
    bitwise_and as bitwise_and,
    bitwise_invert as bitwise_invert,
    bitwise_left_shift as bitwise_left_shift,
    bitwise_not as bitwise_not,
    bitwise_or as bitwise_or,
    bitwise_right_shift as bitwise_right_shift,
    bitwise_xor as bitwise_xor,
    blackman as blackman,
    broadcast_to as broadcast_to,
    cbrt as cbrt,
    ceil as ceil,
    clip as clip,
    concatenate as concatenate,
    conj as conj,
    conjugate as conjugate,
    copy as copy,
    corrcoef as corrcoef,
    correlate as correlate,
    cos as cos,
    cosh as cosh,
    count_nonzero as count_nonzero,
    cross as cross,
    cumprod as cumprod,
    cumsum as cumsum,
    deg2rad as deg2rad,
    diag as diag,
    diagflat as diagflat,
    diagonal as diagonal,
    diff as diff,
    digitize as digitize,
    divide as divide,
    divide_no_nan as divide_no_nan,
    dot as dot,
    einsum as einsum,
    empty as empty,
    empty_like as empty_like,
    equal as equal,
    exp as exp,
    exp2 as exp2,
    expand_dims as expand_dims,
    expm1 as expm1,
    eye as eye,
    flip as flip,
    floor as floor,
    floor_divide as floor_divide,
    full as full,
    full_like as full_like,
    gcd as gcd,
    get_item as get_item,
    greater as greater,
    greater_equal as greater_equal,
    hamming as hamming,
    hanning as hanning,
    heaviside as heaviside,
    histogram as histogram,
    hstack as hstack,
    hypot as hypot,
    identity as identity,
    imag as imag,
    inner as inner,
    isclose as isclose,
    isfinite as isfinite,
    isin as isin,
    isinf as isinf,
    isnan as isnan,
    isneginf as isneginf,
    isposinf as isposinf,
    isreal as isreal,
    kaiser as kaiser,
    kron as kron,
    lcm as lcm,
    ldexp as ldexp,
    left_shift as left_shift,
    less as less,
    less_equal as less_equal,
    linspace as linspace,
    log as log,
    log10 as log10,
    log1p as log1p,
    log2 as log2,
    logaddexp as logaddexp,
    logaddexp2 as logaddexp2,
    logical_and as logical_and,
    logical_not as logical_not,
    logical_or as logical_or,
    logical_xor as logical_xor,
    logspace as logspace,
    matmul as matmul,
    max as max,
    maximum as maximum,
    mean as mean,
    median as median,
    meshgrid as meshgrid,
    min as min,
    minimum as minimum,
    mod as mod,
    moveaxis as moveaxis,
    multiply as multiply,
    nan_to_num as nan_to_num,
    ndim as ndim,
    negative as negative,
    nonzero as nonzero,
    not_equal as not_equal,
    ones as ones,
    ones_like as ones_like,
    outer as outer,
    pad as pad,
    power as power,
    prod as prod,
    quantile as quantile,
    ravel as ravel,
    real as real,
    reciprocal as reciprocal,
    repeat as repeat,
    reshape as reshape,
    right_shift as right_shift,
    roll as roll,
    rot90 as rot90,
    round as round,
    searchsorted as searchsorted,
    select as select,
    sign as sign,
    signbit as signbit,
    sin as sin,
    sinh as sinh,
    size as size,
    slogdet as slogdet,
    sort as sort,
    split as split,
    sqrt as sqrt,
    square as square,
    squeeze as squeeze,
    stack as stack,
    std as std,
    subtract as subtract,
    sum as sum,
    swapaxes as swapaxes,
    take as take,
    take_along_axis as take_along_axis,
    tan as tan,
    tanh as tanh,
    tensordot as tensordot,
    tile as tile,
    trace as trace,
    transpose as transpose,
    trapezoid as trapezoid,
    tri as tri,
    tril as tril,
    triu as triu,
    true_divide as true_divide,
    trunc as trunc,
    unravel_index as unravel_index,
    vander as vander,
    var as var,
    vdot as vdot,
    vectorize as vectorize,
    view as view,
    vstack as vstack,
    where as where,
    zeros as zeros,
    zeros_like as zeros_like,
)

# Re-exported nn operations at top level
from .nn import (
    adaptive_average_pool as adaptive_average_pool,
    adaptive_max_pool as adaptive_max_pool,
    average_pool as average_pool,
    batch_normalization as batch_normalization,
    binary_crossentropy as binary_crossentropy,
    categorical_crossentropy as categorical_crossentropy,
    celu as celu,
    conv as conv,
    conv_transpose as conv_transpose,
    ctc_decode as ctc_decode,
    ctc_loss as ctc_loss,
    depthwise_conv as depthwise_conv,
    dot_product_attention as dot_product_attention,
    elu as elu,
    gelu as gelu,
    glu as glu,
    hard_shrink as hard_shrink,
    hard_sigmoid as hard_sigmoid,
    hard_silu as hard_silu,
    hard_swish as hard_swish,
    hard_tanh as hard_tanh,
    layer_normalization as layer_normalization,
    leaky_relu as leaky_relu,
    log_sigmoid as log_sigmoid,
    log_softmax as log_softmax,
    max_pool as max_pool,
    moments as moments,
    multi_hot as multi_hot,
    normalize as normalize,
    one_hot as one_hot,
    polar as polar,
    psnr as psnr,
    relu as relu,
    relu6 as relu6,
    rms_normalization as rms_normalization,
    selu as selu,
    separable_conv as separable_conv,
    sigmoid as sigmoid,
    silu as silu,
    soft_shrink as soft_shrink,
    softmax as softmax,
    softplus as softplus,
    softsign as softsign,
    sparse_categorical_crossentropy as sparse_categorical_crossentropy,
    sparse_plus as sparse_plus,
    sparse_sigmoid as sparse_sigmoid,
    sparsemax as sparsemax,
    squareplus as squareplus,
    swish as swish,
    tanh_shrink as tanh_shrink,
    threshold as threshold,
    unfold as unfold,
)

# Re-exported linalg operations at top level
from .linalg import (
    cholesky as cholesky,
    cholesky_inverse as cholesky_inverse,
    det as det,
    eig as eig,
    eigh as eigh,
    inv as inv,
    lstsq as lstsq,
    lu_factor as lu_factor,
    qr as qr,
    solve as solve,
    solve_triangular as solve_triangular,
    svd as svd,
)

# Core operations defined in ops/__init__.py
def cast(x: Any, dtype: Any) -> Any: ...
def cond(pred: Any, true_fn: Callable, false_fn: Callable) -> Any: ...
def convert_to_numpy(x: Any) -> Any: ...
def convert_to_tensor(x: Any, dtype: Any = ...) -> Any: ...
def custom_gradient(f: Callable) -> Callable: ...
def dtype(x: Any) -> Any: ...
def erf(x: Any) -> Any: ...
def erfinv(x: Any) -> Any: ...
def extract_sequences(x: Any, sequence_length: int, sequence_stride: int = ...) -> Any: ...
def fft(x: Any) -> Any: ...
def fft2(x: Any) -> Any: ...
def fori_loop(lower: int, upper: int, body_fn: Callable, init_val: Any) -> Any: ...
def ifft2(x: Any) -> Any: ...
def in_top_k(targets: Any, predictions: Any, k: int) -> Any: ...
def irfft(x: Any, fft_length: Optional[int] = ...) -> Any: ...
def is_tensor(x: Any) -> bool: ...
def istft(x: Any, sequence_length: int, sequence_stride: int, fft_length: Optional[int] = ..., window: str = ..., center: bool = ...) -> Any: ...
def jvp(primals: Any, tangents: Any, fn: Callable) -> Any: ...
def logdet(x: Any) -> Any: ...
def logsumexp(x: Any, axis: Optional[int] = ..., keepdims: bool = ...) -> Any: ...
def map(f: Callable, xs: Any) -> Any: ...
def norm(x: Any, ord: Any = ..., axis: Any = ..., keepdims: bool = ...) -> Any: ...
def rearrange(x: Any, pattern: str, **axes_lengths: Any) -> Any: ...
def rfft(x: Any, fft_length: Optional[int] = ...) -> Any: ...
def rsqrt(x: Any) -> Any: ...
def saturate_cast(x: Any, dtype: Any) -> Any: ...
def scan(f: Callable, init: Any, xs: Any, length: Optional[int] = ..., reverse: bool = ..., unroll: int = ...) -> Any: ...
def scatter(indices: Any, values: Any, shape: Any) -> Any: ...
def scatter_update(inputs: Any, indices: Any, updates: Any) -> Any: ...
def segment_max(data: Any, segment_ids: Any, num_segments: Optional[int] = ..., sorted: bool = ...) -> Any: ...
def segment_sum(data: Any, segment_ids: Any, num_segments: Optional[int] = ..., sorted: bool = ...) -> Any: ...
def shape(x: Any) -> Any: ...
def slice(x: Any, start: Any, shape: Any) -> Any: ...
def slice_update(x: Any, start_indices: Any, values: Any) -> Any: ...
def stft(x: Any, sequence_length: int, sequence_stride: int, fft_length: Optional[int] = ..., window: str = ..., center: bool = ...) -> Any: ...
def stop_gradient(x: Any) -> Any: ...
def switch(index: Any, branches: Any, *operands: Any) -> Any: ...
def top_k(x: Any, k: int, sorted: bool = ...) -> Any: ...
def unstack(x: Any, num: Optional[int] = ..., axis: int = ...) -> Any: ...
def vectorized_map(fn: Callable, elems: Any) -> Any: ...
def view_as_complex(x: Any) -> Any: ...
def view_as_real(x: Any) -> Any: ...
def while_loop(cond: Callable, body: Callable, loop_vars: Any, maximum_iterations: Optional[int] = ...) -> Any: ...
def associative_scan(f: Callable, elems: Any, axis: int = ..., reverse: bool = ...) -> Any: ...
