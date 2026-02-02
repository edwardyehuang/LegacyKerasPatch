"""
Neural network operations for Keras 2 compatibility.

This module provides keras.ops.nn compatible operations by wrapping TensorFlow functions.
"""

import numpy as np
import tensorflow as tf


def adaptive_average_pool(inputs, output_size, data_format="channels_last"):
    """Adaptive average pooling operation."""
    input_shape = tf.shape(inputs)
    ndim = len(inputs.shape)

    if ndim == 3:
        # 1D pooling
        if data_format == "channels_last":
            input_length = input_shape[1]
        else:
            input_length = input_shape[2]

        if isinstance(output_size, (list, tuple)):
            output_size = output_size[0]

        pool_size = input_length // output_size
        return tf.nn.avg_pool1d(inputs, pool_size, pool_size, "VALID", data_format.upper())
    elif ndim == 4:
        # 2D pooling
        if data_format == "channels_last":
            h, w = input_shape[1], input_shape[2]
        else:
            h, w = input_shape[2], input_shape[3]

        if isinstance(output_size, int):
            output_size = (output_size, output_size)

        pool_h = h // output_size[0]
        pool_w = w // output_size[1]
        return tf.nn.avg_pool2d(inputs, (pool_h, pool_w), (pool_h, pool_w), "VALID", data_format.upper())
    elif ndim == 5:
        # 3D pooling
        if data_format == "channels_last":
            d, h, w = input_shape[1], input_shape[2], input_shape[3]
        else:
            d, h, w = input_shape[2], input_shape[3], input_shape[4]

        if isinstance(output_size, int):
            output_size = (output_size, output_size, output_size)

        pool_d = d // output_size[0]
        pool_h = h // output_size[1]
        pool_w = w // output_size[2]
        return tf.nn.avg_pool3d(inputs, (pool_d, pool_h, pool_w), (pool_d, pool_h, pool_w), "VALID", data_format.upper())
    else:
        raise ValueError(f"Invalid input dimension: {ndim}")


def adaptive_max_pool(inputs, output_size, data_format="channels_last"):
    """Adaptive max pooling operation."""
    input_shape = tf.shape(inputs)
    ndim = len(inputs.shape)

    if ndim == 3:
        # 1D pooling
        if data_format == "channels_last":
            input_length = input_shape[1]
        else:
            input_length = input_shape[2]

        if isinstance(output_size, (list, tuple)):
            output_size = output_size[0]

        pool_size = input_length // output_size
        return tf.nn.max_pool1d(inputs, pool_size, pool_size, "VALID", data_format.upper())
    elif ndim == 4:
        # 2D pooling
        if data_format == "channels_last":
            h, w = input_shape[1], input_shape[2]
        else:
            h, w = input_shape[2], input_shape[3]

        if isinstance(output_size, int):
            output_size = (output_size, output_size)

        pool_h = h // output_size[0]
        pool_w = w // output_size[1]
        return tf.nn.max_pool2d(inputs, (pool_h, pool_w), (pool_h, pool_w), "VALID", data_format.upper())
    elif ndim == 5:
        # 3D pooling
        if data_format == "channels_last":
            d, h, w = input_shape[1], input_shape[2], input_shape[3]
        else:
            d, h, w = input_shape[2], input_shape[3], input_shape[4]

        if isinstance(output_size, int):
            output_size = (output_size, output_size, output_size)

        pool_d = d // output_size[0]
        pool_h = h // output_size[1]
        pool_w = w // output_size[2]
        return tf.nn.max_pool3d(inputs, (pool_d, pool_h, pool_w), (pool_d, pool_h, pool_w), "VALID", data_format.upper())
    else:
        raise ValueError(f"Invalid input dimension: {ndim}")


def average_pool(inputs, pool_size, strides=None, padding="valid", data_format="channels_last"):
    """Average pooling operation."""
    if strides is None:
        strides = pool_size
    padding = padding.upper()
    data_format = "NHWC" if data_format == "channels_last" else "NCHW"
    ndim = len(inputs.shape)
    
    if ndim == 3:
        return tf.nn.avg_pool1d(inputs, pool_size, strides, padding, data_format)
    elif ndim == 4:
        return tf.nn.avg_pool2d(inputs, pool_size, strides, padding, data_format)
    elif ndim == 5:
        return tf.nn.avg_pool3d(inputs, pool_size, strides, padding, data_format)
    else:
        raise ValueError(f"Invalid input dimension: {ndim}")


def batch_normalization(x, mean, variance, axis, offset=None, scale=None, epsilon=1e-3):
    """Batch normalization operation."""
    return tf.nn.batch_normalization(x, mean, variance, offset, scale, epsilon)


def binary_crossentropy(target, output, from_logits=False):
    """Binary crossentropy loss."""
    if from_logits:
        return tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=output)
    else:
        output = tf.clip_by_value(output, 1e-7, 1 - 1e-7)
        return -target * tf.math.log(output) - (1 - target) * tf.math.log(1 - output)


def categorical_crossentropy(target, output, from_logits=False, axis=-1):
    """Categorical crossentropy loss."""
    if from_logits:
        return tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=output, axis=axis)
    else:
        output = tf.clip_by_value(output, 1e-7, 1 - 1e-7)
        return -tf.reduce_sum(target * tf.math.log(output), axis=axis)


def celu(x, alpha=1.0):
    """Continuously-differentiable exponential linear unit."""
    return tf.maximum(x, 0.0) + tf.minimum(0.0, alpha * (tf.exp(x / alpha) - 1))


def conv(inputs, kernel, strides=1, padding="valid", data_format="channels_last", dilation_rate=1):
    """Convolution operation."""
    padding = padding.upper()
    ndim = len(inputs.shape) - 2  # subtract batch and channel dims
    
    if data_format == "channels_last":
        data_format_tf = "NHWC" if ndim == 2 else ("NDHWC" if ndim == 3 else "NWC")
    else:
        data_format_tf = "NCHW" if ndim == 2 else ("NCDHW" if ndim == 3 else "NCW")
    
    if ndim == 1:
        return tf.nn.conv1d(inputs, kernel, strides, padding, data_format=data_format_tf, dilations=dilation_rate)
    elif ndim == 2:
        return tf.nn.conv2d(inputs, kernel, strides, padding, data_format=data_format_tf, dilations=dilation_rate)
    elif ndim == 3:
        return tf.nn.conv3d(inputs, kernel, strides, padding, data_format=data_format_tf, dilations=dilation_rate)
    else:
        raise ValueError(f"Invalid input dimension: {ndim + 2}")


def conv_transpose(inputs, kernel, strides, padding="valid", output_padding=None, data_format="channels_last", dilation_rate=1):
    """Transposed convolution operation."""
    padding = padding.upper()
    ndim = len(inputs.shape) - 2
    
    if ndim == 1:
        return tf.nn.conv1d_transpose(inputs, kernel, output_shape=None, strides=strides, padding=padding, data_format=data_format.upper(), dilations=dilation_rate)
    elif ndim == 2:
        return tf.nn.conv2d_transpose(inputs, kernel, output_shape=None, strides=strides, padding=padding, data_format="NHWC" if data_format == "channels_last" else "NCHW", dilations=dilation_rate)
    elif ndim == 3:
        return tf.nn.conv3d_transpose(inputs, kernel, output_shape=None, strides=strides, padding=padding, data_format="NDHWC" if data_format == "channels_last" else "NCDHW", dilations=dilation_rate)
    else:
        raise ValueError(f"Invalid input dimension: {ndim + 2}")


def ctc_decode(inputs, sequence_lengths, strategy="greedy", beam_width=100, top_paths=1, merge_repeated=True, mask_index=None):
    """CTC decoding."""
    inputs = tf.transpose(inputs, [1, 0, 2])  # (batch, time, classes) -> (time, batch, classes)
    
    if strategy == "greedy":
        decoded, log_probs = tf.nn.ctc_greedy_decoder(inputs, sequence_lengths, merge_repeated=merge_repeated)
    else:
        decoded, log_probs = tf.nn.ctc_beam_search_decoder(inputs, sequence_lengths, beam_width=beam_width, top_paths=top_paths)
    
    return decoded, log_probs


def ctc_loss(target, output, target_length, output_length, mask_index=0):
    """CTC loss."""
    return tf.nn.ctc_loss(
        labels=target,
        logits=output,
        label_length=target_length,
        logit_length=output_length,
        logits_time_major=False,
        blank_index=mask_index
    )


def depthwise_conv(inputs, kernel, strides=1, padding="valid", data_format="channels_last", dilation_rate=1):
    """Depthwise convolution operation."""
    padding = padding.upper()
    if data_format == "channels_last":
        strides = [1, strides, strides, 1] if isinstance(strides, int) else [1] + list(strides) + [1]
        data_format_tf = "NHWC"
    else:
        strides = [1, 1, strides, strides] if isinstance(strides, int) else [1, 1] + list(strides)
        data_format_tf = "NCHW"
    
    if isinstance(dilation_rate, int):
        dilation_rate = [dilation_rate, dilation_rate]
    
    return tf.nn.depthwise_conv2d(inputs, kernel, strides, padding, data_format=data_format_tf, dilations=dilation_rate)


def dot_product_attention(query, key, value, bias=None, mask=None, scale=None, is_causal=False, flash_attention=None):
    """Scaled dot product attention."""
    if scale is None:
        scale = 1.0 / tf.math.sqrt(tf.cast(tf.shape(key)[-1], query.dtype))
    
    scores = tf.matmul(query, key, transpose_b=True) * scale
    
    if bias is not None:
        scores = scores + bias
    
    if mask is not None:
        scores = tf.where(mask, scores, tf.constant(-1e9, dtype=scores.dtype))
    
    if is_causal:
        seq_len = tf.shape(scores)[-1]
        causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        scores = tf.where(causal_mask == 1, scores, tf.constant(-1e9, dtype=scores.dtype))
    
    weights = tf.nn.softmax(scores, axis=-1)
    return tf.matmul(weights, value)


def elu(x, alpha=1.0):
    """Exponential linear unit."""
    return tf.nn.elu(x) if alpha == 1.0 else tf.where(x > 0, x, alpha * (tf.exp(x) - 1))


def gelu(x, approximate=True):
    """Gaussian error linear unit."""
    if approximate:
        return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))
    else:
        return x * 0.5 * (1.0 + tf.math.erf(x / tf.sqrt(2.0)))


def glu(x, axis=-1):
    """Gated linear unit."""
    a, b = tf.split(x, 2, axis=axis)
    return a * tf.sigmoid(b)


def hard_shrink(x, threshold=0.5):
    """Hard shrink activation function."""
    return tf.where(tf.abs(x) > threshold, x, tf.zeros_like(x))


def hard_sigmoid(x):
    """Hard sigmoid activation function."""
    return tf.clip_by_value(x / 6 + 0.5, 0, 1)


def hard_silu(x):
    """Hard SiLU (Swish) activation function."""
    return x * hard_sigmoid(x)


def hard_swish(x):
    """Hard Swish activation function (alias for hard_silu)."""
    return hard_silu(x)


def hard_tanh(x):
    """Hard tanh activation function."""
    return tf.clip_by_value(x, -1, 1)


def layer_normalization(x, scale=None, offset=None, axis=-1, epsilon=1e-5):
    """Layer normalization operation."""
    mean, variance = tf.nn.moments(x, axes=[axis], keepdims=True)
    normalized = (x - mean) / tf.sqrt(variance + epsilon)
    
    if scale is not None:
        normalized = normalized * scale
    if offset is not None:
        normalized = normalized + offset
    
    return normalized


def leaky_relu(x, negative_slope=0.2):
    """Leaky ReLU activation function."""
    return tf.nn.leaky_relu(x, alpha=negative_slope)


def log_sigmoid(x):
    """Log sigmoid activation function."""
    return tf.math.log_sigmoid(x)


def log_softmax(x, axis=-1):
    """Log softmax activation function."""
    return tf.nn.log_softmax(x, axis=axis)


def max_pool(inputs, pool_size, strides=None, padding="valid", data_format="channels_last"):
    """Max pooling operation."""
    if strides is None:
        strides = pool_size
    padding = padding.upper()
    data_format = "NHWC" if data_format == "channels_last" else "NCHW"
    ndim = len(inputs.shape)
    
    if ndim == 3:
        return tf.nn.max_pool1d(inputs, pool_size, strides, padding, data_format)
    elif ndim == 4:
        return tf.nn.max_pool2d(inputs, pool_size, strides, padding, data_format)
    elif ndim == 5:
        return tf.nn.max_pool3d(inputs, pool_size, strides, padding, data_format)
    else:
        raise ValueError(f"Invalid input dimension: {ndim}")


def moments(x, axes, keepdims=False, synchronized=False):
    """Calculate mean and variance."""
    return tf.nn.moments(x, axes, keepdims=keepdims)


def multi_hot(inputs, num_classes, axis=-1, dtype=None, sparse=False):
    """Multi-hot encoding."""
    dtype = dtype or tf.float32
    return tf.reduce_sum(tf.one_hot(inputs, num_classes, dtype=dtype), axis=axis)


def normalize(x, axis=-1, order=2, epsilon=1e-12):
    """L-p normalize along axis."""
    if order == 2:
        return tf.nn.l2_normalize(x, axis=axis, epsilon=epsilon)
    else:
        norm = tf.pow(tf.reduce_sum(tf.pow(tf.abs(x), order), axis=axis, keepdims=True), 1.0 / order)
        return x / tf.maximum(norm, epsilon)


def one_hot(x, num_classes, axis=-1, dtype=None, sparse=False):
    """One-hot encoding."""
    dtype = dtype or tf.float32
    return tf.one_hot(x, num_classes, axis=axis, dtype=dtype)


def polar(x):
    """Polar activation function."""
    return tf.tanh(x / 2)


def psnr(x1, x2, max_val):
    """Peak signal-to-noise ratio."""
    return tf.image.psnr(x1, x2, max_val)


def relu(x):
    """Rectified linear unit."""
    return tf.nn.relu(x)


def relu6(x):
    """ReLU6 activation function."""
    return tf.nn.relu6(x)


def rms_normalization(x, scale=None, offset=None, axis=-1, epsilon=1e-5):
    """RMS normalization operation."""
    rms = tf.sqrt(tf.reduce_mean(tf.square(x), axis=axis, keepdims=True) + epsilon)
    normalized = x / rms
    
    if scale is not None:
        normalized = normalized * scale
    if offset is not None:
        normalized = normalized + offset
    
    return normalized


def selu(x):
    """Scaled exponential linear unit."""
    return tf.nn.selu(x)


def separable_conv(inputs, depthwise_kernel, pointwise_kernel, strides=1, padding="valid", data_format="channels_last", dilation_rate=1):
    """Separable convolution operation."""
    padding = padding.upper()
    if data_format == "channels_last":
        strides = [1, strides, strides, 1] if isinstance(strides, int) else [1] + list(strides) + [1]
    else:
        strides = [1, 1, strides, strides] if isinstance(strides, int) else [1, 1] + list(strides)
    
    if isinstance(dilation_rate, int):
        dilation_rate = [dilation_rate, dilation_rate]
    
    return tf.nn.separable_conv2d(
        inputs, depthwise_kernel, pointwise_kernel, strides, padding,
        data_format="NHWC" if data_format == "channels_last" else "NCHW",
        dilations=dilation_rate
    )


def sigmoid(x):
    """Sigmoid activation function."""
    return tf.nn.sigmoid(x)


def silu(x):
    """SiLU (Swish) activation function."""
    return tf.nn.silu(x)


def soft_shrink(x, threshold=0.5):
    """Soft shrink activation function."""
    return tf.sign(x) * tf.maximum(tf.abs(x) - threshold, 0)


def softmax(x, axis=-1):
    """Softmax activation function."""
    return tf.nn.softmax(x, axis=axis)


def softplus(x):
    """Softplus activation function."""
    return tf.nn.softplus(x)


def softsign(x):
    """Softsign activation function."""
    return tf.nn.softsign(x)


def sparse_categorical_crossentropy(target, output, from_logits=False, axis=-1):
    """Sparse categorical crossentropy loss."""
    if from_logits:
        return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(target, tf.int32), logits=output)
    else:
        output = tf.clip_by_value(output, 1e-7, 1 - 1e-7)
        target = tf.cast(target, tf.int32)
        return -tf.reduce_sum(tf.one_hot(target, tf.shape(output)[-1]) * tf.math.log(output), axis=axis)


def sparse_plus(x):
    """Sparse plus activation function."""
    return tf.where(x <= -1, tf.zeros_like(x), tf.where(x >= 1, x, 0.25 * tf.square(x + 1)))


def sparse_sigmoid(x):
    """Sparse sigmoid activation function."""
    return tf.clip_by_value(0.25 * (x + 1), 0, 1)


def sparsemax(x, axis=-1):
    """Sparsemax activation function."""
    # Simple implementation
    x_sorted = tf.sort(x, axis=axis, direction='DESCENDING')
    cum_sum = tf.cumsum(x_sorted, axis=axis)
    k = tf.range(1, tf.shape(x)[axis] + 1, dtype=x.dtype)
    
    # Find threshold
    threshold = (cum_sum - 1) / k
    support = tf.cast(x_sorted > threshold, x.dtype)
    k_z = tf.reduce_sum(support, axis=axis, keepdims=True)
    tau = (tf.reduce_sum(x_sorted * support, axis=axis, keepdims=True) - 1) / k_z
    
    return tf.maximum(x - tau, 0)


def squareplus(x, b=4):
    """Squareplus activation function."""
    return 0.5 * (x + tf.sqrt(tf.square(x) + b))


def swish(x):
    """Swish activation function (alias for silu)."""
    return silu(x)


def tanh_shrink(x):
    """Tanh shrink activation function."""
    return x - tf.tanh(x)


def threshold(x, threshold_value, value):
    """Threshold activation function."""
    return tf.where(x > threshold_value, x, value)


def unfold(inputs, kernel_size, strides=1, dilation_rate=1, padding="valid", data_format="channels_last"):
    """Extract patches from input tensor."""
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]
    if isinstance(strides, int):
        strides = [1, strides, strides, 1]
    else:
        strides = [1] + list(strides) + [1]
    if isinstance(dilation_rate, int):
        dilation_rate = [1, dilation_rate, dilation_rate, 1]
    else:
        dilation_rate = [1] + list(dilation_rate) + [1]
    
    sizes = [1, kernel_size[0], kernel_size[1], 1]
    
    if data_format == "channels_first":
        inputs = tf.transpose(inputs, [0, 2, 3, 1])
    
    patches = tf.image.extract_patches(inputs, sizes, strides, dilation_rate, padding.upper())
    
    return patches
