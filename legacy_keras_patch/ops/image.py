"""
Image operations for Keras 2 compatibility.

This module provides keras.ops.image compatible operations by wrapping TensorFlow functions.
"""

import tensorflow as tf


def affine_transform(images, transform, interpolation="bilinear", fill_mode="constant", fill_value=0.0, data_format="channels_last"):
    """Apply an affine transformation to images."""
    if data_format == "channels_first":
        images = tf.transpose(images, [0, 2, 3, 1])
    
    result = tf.raw_ops.ImageProjectiveTransformV3(
        images=images,
        transforms=transform,
        output_shape=tf.shape(images)[1:3],
        interpolation=interpolation.upper(),
        fill_mode=fill_mode.upper(),
        fill_value=fill_value
    )
    
    if data_format == "channels_first":
        result = tf.transpose(result, [0, 3, 1, 2])
    
    return result


def crop_images(images, top_cropping, left_cropping, bottom_cropping, right_cropping, target_height=None, target_width=None, data_format="channels_last"):
    """Crop images to a specified size."""
    if data_format == "channels_first":
        images = tf.transpose(images, [0, 2, 3, 1])
    
    shape = tf.shape(images)
    height = shape[1]
    width = shape[2]
    
    new_height = height - top_cropping - bottom_cropping
    new_width = width - left_cropping - right_cropping
    
    if target_height is not None:
        new_height = target_height
    if target_width is not None:
        new_width = target_width
    
    result = tf.image.crop_to_bounding_box(images, top_cropping, left_cropping, new_height, new_width)
    
    if data_format == "channels_first":
        result = tf.transpose(result, [0, 3, 1, 2])
    
    return result


def elastic_transform(images, alpha, sigma, interpolation="bilinear", fill_mode="constant", fill_value=0.0, seed=None, data_format="channels_last"):
    """Apply elastic transformation to images."""
    if data_format == "channels_first":
        images = tf.transpose(images, [0, 2, 3, 1])
    
    shape = tf.shape(images)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]
    
    # Generate random displacement fields
    if seed is not None:
        tf.random.set_seed(seed)
    
    dx = tf.random.uniform([batch_size, height, width], -1, 1) * alpha
    dy = tf.random.uniform([batch_size, height, width], -1, 1) * alpha
    
    # Apply Gaussian smoothing (simplified)
    kernel_size = int(sigma * 6) | 1  # Ensure odd
    kernel = tf.cast(tf.range(kernel_size), tf.float32) - kernel_size // 2
    kernel = tf.exp(-0.5 * (kernel / sigma) ** 2)
    kernel = kernel / tf.reduce_sum(kernel)
    kernel = kernel[:, tf.newaxis] * kernel[tf.newaxis, :]
    kernel = kernel[:, :, tf.newaxis, tf.newaxis]
    
    dx = tf.nn.depthwise_conv2d(dx[:, :, :, tf.newaxis], kernel, [1, 1, 1, 1], "SAME")[:, :, :, 0]
    dy = tf.nn.depthwise_conv2d(dy[:, :, :, tf.newaxis], kernel, [1, 1, 1, 1], "SAME")[:, :, :, 0]
    
    # Create grid and apply displacements
    x = tf.range(width, dtype=tf.float32)
    y = tf.range(height, dtype=tf.float32)
    X, Y = tf.meshgrid(x, y)
    X = tf.tile(X[tf.newaxis, :, :], [batch_size, 1, 1])
    Y = tf.tile(Y[tf.newaxis, :, :], [batch_size, 1, 1])
    
    X = X + dx
    Y = Y + dy
    
    # Create warp coordinates
    coords = tf.stack([Y, X], axis=-1)
    
    # Use dense_image_warp for interpolation
    result = _dense_image_warp(images, coords - tf.stack([Y - dy, X - dx], axis=-1), interpolation)
    
    if data_format == "channels_first":
        result = tf.transpose(result, [0, 3, 1, 2])
    
    return result


def _dense_image_warp(image, flow, interpolation="bilinear"):
    """Warp image using flow field (helper function)."""
    batch_size, height, width, channels = tf.unstack(tf.shape(image))
    
    # Get pixel coordinates
    y = tf.range(height, dtype=tf.float32)
    x = tf.range(width, dtype=tf.float32)
    X, Y = tf.meshgrid(x, y)
    X = tf.tile(X[tf.newaxis, :, :], [batch_size, 1, 1])
    Y = tf.tile(Y[tf.newaxis, :, :], [batch_size, 1, 1])
    
    # Apply flow
    new_Y = Y + flow[:, :, :, 0]
    new_X = X + flow[:, :, :, 1]
    
    # Bilinear interpolation
    new_X = tf.clip_by_value(new_X, 0, tf.cast(width - 1, tf.float32))
    new_Y = tf.clip_by_value(new_Y, 0, tf.cast(height - 1, tf.float32))
    
    x0 = tf.floor(new_X)
    x1 = x0 + 1
    y0 = tf.floor(new_Y)
    y1 = y0 + 1
    
    x0 = tf.clip_by_value(x0, 0, tf.cast(width - 1, tf.float32))
    x1 = tf.clip_by_value(x1, 0, tf.cast(width - 1, tf.float32))
    y0 = tf.clip_by_value(y0, 0, tf.cast(height - 1, tf.float32))
    y1 = tf.clip_by_value(y1, 0, tf.cast(height - 1, tf.float32))
    
    x0i = tf.cast(x0, tf.int32)
    x1i = tf.cast(x1, tf.int32)
    y0i = tf.cast(y0, tf.int32)
    y1i = tf.cast(y1, tf.int32)
    
    batch_idx = tf.tile(tf.range(batch_size)[:, tf.newaxis, tf.newaxis], [1, height, width])
    
    def gather_pixel(y_idx, x_idx):
        indices = tf.stack([batch_idx, y_idx, x_idx], axis=-1)
        return tf.gather_nd(image, indices)
    
    Ia = gather_pixel(y0i, x0i)
    Ib = gather_pixel(y1i, x0i)
    Ic = gather_pixel(y0i, x1i)
    Id = gather_pixel(y1i, x1i)
    
    wa = (x1 - new_X) * (y1 - new_Y)
    wb = (x1 - new_X) * (new_Y - y0)
    wc = (new_X - x0) * (y1 - new_Y)
    wd = (new_X - x0) * (new_Y - y0)
    
    wa = wa[:, :, :, tf.newaxis]
    wb = wb[:, :, :, tf.newaxis]
    wc = wc[:, :, :, tf.newaxis]
    wd = wd[:, :, :, tf.newaxis]
    
    return wa * Ia + wb * Ib + wc * Ic + wd * Id


def extract_patches(images, size, strides=None, dilation_rate=1, padding="valid", data_format="channels_last"):
    """Extract patches from images."""
    if data_format == "channels_first":
        images = tf.transpose(images, [0, 2, 3, 1])
    
    if isinstance(size, int):
        size = [1, size, size, 1]
    else:
        size = [1, size[0], size[1], 1]
    
    if strides is None:
        strides = size
    elif isinstance(strides, int):
        strides = [1, strides, strides, 1]
    else:
        strides = [1, strides[0], strides[1], 1]
    
    if isinstance(dilation_rate, int):
        rates = [1, dilation_rate, dilation_rate, 1]
    else:
        rates = [1, dilation_rate[0], dilation_rate[1], 1]
    
    result = tf.image.extract_patches(images, size, strides, rates, padding.upper())
    
    return result


def extract_patches_3d(images, size, strides=None, dilation_rate=1, padding="valid", data_format="channels_last"):
    """Extract 3D patches from volumes."""
    if data_format == "channels_first":
        images = tf.transpose(images, [0, 2, 3, 4, 1])
    
    if isinstance(size, int):
        size = [size, size, size]
    
    if strides is None:
        strides = size
    elif isinstance(strides, int):
        strides = [strides, strides, strides]
    
    # Use extract_volume_patches if available, otherwise implement manually
    result = tf.extract_volume_patches(
        images,
        ksizes=[1] + list(size) + [1],
        strides=[1] + list(strides) + [1],
        padding=padding.upper()
    )
    
    return result


def gaussian_blur(images, kernel_size, sigma, data_format="channels_last"):
    """Apply Gaussian blur to images."""
    if data_format == "channels_first":
        images = tf.transpose(images, [0, 2, 3, 1])
    
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]
    if isinstance(sigma, (int, float)):
        sigma = [sigma, sigma]
    
    # Create Gaussian kernel
    def make_gaussian_kernel(size, sigma):
        x = tf.range(size, dtype=tf.float32) - (size - 1) / 2
        kernel = tf.exp(-0.5 * (x / sigma) ** 2)
        return kernel / tf.reduce_sum(kernel)
    
    kernel_h = make_gaussian_kernel(kernel_size[0], sigma[0])
    kernel_w = make_gaussian_kernel(kernel_size[1], sigma[1])
    
    kernel_2d = kernel_h[:, tf.newaxis] * kernel_w[tf.newaxis, :]
    
    channels = tf.shape(images)[-1]
    kernel = kernel_2d[:, :, tf.newaxis, tf.newaxis]
    kernel = tf.tile(kernel, [1, 1, channels, 1])
    
    result = tf.nn.depthwise_conv2d(images, kernel, [1, 1, 1, 1], "SAME")
    
    if data_format == "channels_first":
        result = tf.transpose(result, [0, 3, 1, 2])
    
    return result


def hsv_to_rgb(images, data_format="channels_last"):
    """Convert HSV images to RGB."""
    if data_format == "channels_first":
        images = tf.transpose(images, [0, 2, 3, 1])
    
    result = tf.image.hsv_to_rgb(images)
    
    if data_format == "channels_first":
        result = tf.transpose(result, [0, 3, 1, 2])
    
    return result


def map_coordinates(inputs, coordinates, order=1, fill_mode="constant", fill_value=0.0):
    """Map coordinates in an input tensor to output."""
    # This is a simplified implementation for 2D
    ndim = len(inputs.shape)
    
    if ndim == 2:
        # 2D case
        height, width = tf.unstack(tf.shape(inputs))
        y_coords = coordinates[0]
        x_coords = coordinates[1]
        
        if order == 0:
            # Nearest neighbor
            y_idx = tf.clip_by_value(tf.cast(tf.round(y_coords), tf.int32), 0, height - 1)
            x_idx = tf.clip_by_value(tf.cast(tf.round(x_coords), tf.int32), 0, width - 1)
            return tf.gather_nd(inputs, tf.stack([y_idx, x_idx], axis=-1))
        else:
            # Bilinear interpolation
            y_coords = tf.cast(y_coords, tf.float32)
            x_coords = tf.cast(x_coords, tf.float32)
            
            y0 = tf.floor(y_coords)
            y1 = y0 + 1
            x0 = tf.floor(x_coords)
            x1 = x0 + 1
            
            y0 = tf.clip_by_value(y0, 0, tf.cast(height - 1, tf.float32))
            y1 = tf.clip_by_value(y1, 0, tf.cast(height - 1, tf.float32))
            x0 = tf.clip_by_value(x0, 0, tf.cast(width - 1, tf.float32))
            x1 = tf.clip_by_value(x1, 0, tf.cast(width - 1, tf.float32))
            
            y0i = tf.cast(y0, tf.int32)
            y1i = tf.cast(y1, tf.int32)
            x0i = tf.cast(x0, tf.int32)
            x1i = tf.cast(x1, tf.int32)
            
            Ia = tf.gather_nd(inputs, tf.stack([y0i, x0i], axis=-1))
            Ib = tf.gather_nd(inputs, tf.stack([y1i, x0i], axis=-1))
            Ic = tf.gather_nd(inputs, tf.stack([y0i, x1i], axis=-1))
            Id = tf.gather_nd(inputs, tf.stack([y1i, x1i], axis=-1))
            
            wa = (x1 - x_coords) * (y1 - y_coords)
            wb = (x1 - x_coords) * (y_coords - y0)
            wc = (x_coords - x0) * (y1 - y_coords)
            wd = (x_coords - x0) * (y_coords - y0)
            
            Ia = tf.cast(Ia, tf.float32)
            Ib = tf.cast(Ib, tf.float32)
            Ic = tf.cast(Ic, tf.float32)
            Id = tf.cast(Id, tf.float32)
            
            return wa * Ia + wb * Ib + wc * Ic + wd * Id
    else:
        raise NotImplementedError(f"map_coordinates not implemented for {ndim}D inputs")


def pad_images(images, top_padding=0, left_padding=0, bottom_padding=0, right_padding=0, target_height=None, target_width=None, data_format="channels_last"):
    """Pad images."""
    if data_format == "channels_first":
        images = tf.transpose(images, [0, 2, 3, 1])
    
    if target_height is not None or target_width is not None:
        shape = tf.shape(images)
        current_height = shape[1]
        current_width = shape[2]
        
        if target_height is not None:
            total_pad_h = target_height - current_height
            top_padding = total_pad_h // 2
            bottom_padding = total_pad_h - top_padding
        
        if target_width is not None:
            total_pad_w = target_width - current_width
            left_padding = total_pad_w // 2
            right_padding = total_pad_w - left_padding
    
    paddings = [[0, 0], [top_padding, bottom_padding], [left_padding, right_padding], [0, 0]]
    result = tf.pad(images, paddings)
    
    if data_format == "channels_first":
        result = tf.transpose(result, [0, 3, 1, 2])
    
    return result


def perspective_transform(images, transform, interpolation="bilinear", fill_mode="constant", fill_value=0.0, data_format="channels_last"):
    """Apply perspective transformation to images."""
    # Use affine_transform as the base, as TensorFlow's image operations support projective transforms
    return affine_transform(images, transform, interpolation, fill_mode, fill_value, data_format)


def resize(images, size, interpolation="bilinear", antialias=False, crop_to_aspect_ratio=False, pad_to_aspect_ratio=False, fill_mode="constant", fill_value=0.0, data_format="channels_last"):
    """Resize images to a target size."""
    if data_format == "channels_first":
        images = tf.transpose(images, [0, 2, 3, 1])
    
    method_map = {
        "nearest": tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        "bilinear": tf.image.ResizeMethod.BILINEAR,
        "bicubic": tf.image.ResizeMethod.BICUBIC,
        "lanczos3": tf.image.ResizeMethod.LANCZOS3,
        "lanczos5": tf.image.ResizeMethod.LANCZOS5,
        "area": tf.image.ResizeMethod.AREA,
    }
    
    method = method_map.get(interpolation, tf.image.ResizeMethod.BILINEAR)
    
    if isinstance(size, int):
        size = [size, size]
    
    result = tf.image.resize(images, size, method=method, antialias=antialias)
    
    if data_format == "channels_first":
        result = tf.transpose(result, [0, 3, 1, 2])
    
    return result


def rgb_to_grayscale(images, data_format="channels_last"):
    """Convert RGB images to grayscale."""
    if data_format == "channels_first":
        images = tf.transpose(images, [0, 2, 3, 1])
    
    result = tf.image.rgb_to_grayscale(images)
    
    if data_format == "channels_first":
        result = tf.transpose(result, [0, 3, 1, 2])
    
    return result


def rgb_to_hsv(images, data_format="channels_last"):
    """Convert RGB images to HSV."""
    if data_format == "channels_first":
        images = tf.transpose(images, [0, 2, 3, 1])
    
    result = tf.image.rgb_to_hsv(images)
    
    if data_format == "channels_first":
        result = tf.transpose(result, [0, 3, 1, 2])
    
    return result


def scale_and_translate(images, scale, translation, interpolation="bilinear", fill_mode="constant", fill_value=0.0, data_format="channels_last"):
    """Scale and translate images."""
    if data_format == "channels_first":
        images = tf.transpose(images, [0, 2, 3, 1])
    
    # Create affine transformation matrix
    # [scale_x, 0, translate_x]
    # [0, scale_y, translate_y]
    # [0, 0, 1]
    
    batch_size = tf.shape(images)[0]
    
    if isinstance(scale, (int, float)):
        scale = [scale, scale]
    if isinstance(translation, (int, float)):
        translation = [translation, translation]
    
    # TensorFlow uses a flat 8-element transform matrix
    # [a0, a1, a2, b0, b1, b2, c0, c1] representing the transformation
    transforms = tf.stack([
        tf.ones([batch_size]) * scale[1],  # scale x
        tf.zeros([batch_size]),
        tf.ones([batch_size]) * translation[1],  # translate x
        tf.zeros([batch_size]),
        tf.ones([batch_size]) * scale[0],  # scale y
        tf.ones([batch_size]) * translation[0],  # translate y
        tf.zeros([batch_size]),
        tf.zeros([batch_size])
    ], axis=1)
    
    result = affine_transform(images, transforms, interpolation, fill_mode, fill_value, "channels_last")
    
    if data_format == "channels_first":
        result = tf.transpose(result, [0, 3, 1, 2])
    
    return result
