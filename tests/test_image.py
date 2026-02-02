"""Tests for image operations in legacy_keras_patch.ops.image."""

import pytest
import numpy as np
import tensorflow as tf

from legacy_keras_patch.ops import image


class TestImageResize:
    """Test image resize operations."""
    
    def test_resize_bilinear(self):
        """Test bilinear resize."""
        x = tf.random.uniform([1, 8, 8, 3])
        result = image.resize(x, size=(16, 16), interpolation="bilinear")
        assert result.shape == (1, 16, 16, 3)
    
    def test_resize_nearest(self):
        """Test nearest neighbor resize."""
        x = tf.random.uniform([1, 8, 8, 3])
        result = image.resize(x, size=(16, 16), interpolation="nearest")
        assert result.shape == (1, 16, 16, 3)
    
    def test_resize_bicubic(self):
        """Test bicubic resize."""
        x = tf.random.uniform([1, 8, 8, 3])
        result = image.resize(x, size=(16, 16), interpolation="bicubic")
        assert result.shape == (1, 16, 16, 3)
    
    def test_resize_downsample(self):
        """Test downsampling."""
        x = tf.random.uniform([1, 16, 16, 3])
        result = image.resize(x, size=(8, 8))
        assert result.shape == (1, 8, 8, 3)
    
    def test_resize_channels_first(self):
        """Test resize with channels_first format."""
        x = tf.random.uniform([1, 3, 8, 8])
        result = image.resize(x, size=(16, 16), data_format="channels_first")
        assert result.shape == (1, 3, 16, 16)


class TestImageCrop:
    """Test image crop operations."""
    
    def test_crop_images(self):
        """Test image cropping."""
        x = tf.random.uniform([1, 8, 8, 3])
        result = image.crop_images(
            x, 
            top_cropping=1, 
            left_cropping=1, 
            bottom_cropping=1, 
            right_cropping=1
        )
        assert result.shape == (1, 6, 6, 3)
    
    def test_crop_images_channels_first(self):
        """Test image cropping with channels_first format."""
        x = tf.random.uniform([1, 3, 8, 8])
        result = image.crop_images(
            x, 
            top_cropping=1, 
            left_cropping=1, 
            bottom_cropping=1, 
            right_cropping=1,
            data_format="channels_first"
        )
        assert result.shape == (1, 3, 6, 6)


class TestImagePad:
    """Test image padding operations."""
    
    def test_pad_images(self):
        """Test image padding."""
        x = tf.random.uniform([1, 8, 8, 3])
        result = image.pad_images(
            x,
            top_padding=2,
            left_padding=2,
            bottom_padding=2,
            right_padding=2
        )
        assert result.shape == (1, 12, 12, 3)
    
    def test_pad_images_target_size(self):
        """Test image padding with target size."""
        x = tf.random.uniform([1, 8, 8, 3])
        result = image.pad_images(x, target_height=16, target_width=16)
        assert result.shape == (1, 16, 16, 3)
    
    def test_pad_images_channels_first(self):
        """Test image padding with channels_first format."""
        x = tf.random.uniform([1, 3, 8, 8])
        result = image.pad_images(
            x,
            top_padding=2,
            left_padding=2,
            bottom_padding=2,
            right_padding=2,
            data_format="channels_first"
        )
        assert result.shape == (1, 3, 12, 12)


class TestColorConversions:
    """Test color space conversion operations."""
    
    def test_rgb_to_grayscale(self):
        """Test RGB to grayscale conversion."""
        x = tf.random.uniform([1, 8, 8, 3])
        result = image.rgb_to_grayscale(x)
        assert result.shape == (1, 8, 8, 1)
    
    def test_rgb_to_grayscale_channels_first(self):
        """Test RGB to grayscale with channels_first format."""
        x = tf.random.uniform([1, 3, 8, 8])
        result = image.rgb_to_grayscale(x, data_format="channels_first")
        assert result.shape == (1, 1, 8, 8)
    
    def test_rgb_to_hsv(self):
        """Test RGB to HSV conversion."""
        x = tf.random.uniform([1, 8, 8, 3])
        result = image.rgb_to_hsv(x)
        assert result.shape == (1, 8, 8, 3)
        # H, S, V should all be in [0, 1] range (approximately, due to numerical precision)
        assert tf.reduce_all(result >= -0.01).numpy()
        assert tf.reduce_all(result <= 1.01).numpy()
    
    def test_hsv_to_rgb(self):
        """Test HSV to RGB conversion."""
        # Create valid HSV values
        h = tf.random.uniform([1, 8, 8, 1], minval=0, maxval=1)
        s = tf.random.uniform([1, 8, 8, 1], minval=0, maxval=1)
        v = tf.random.uniform([1, 8, 8, 1], minval=0, maxval=1)
        x = tf.concat([h, s, v], axis=-1)
        result = image.hsv_to_rgb(x)
        assert result.shape == (1, 8, 8, 3)
    
    def test_rgb_hsv_roundtrip(self):
        """Test RGB -> HSV -> RGB roundtrip."""
        x = tf.random.uniform([1, 8, 8, 3], minval=0.01, maxval=0.99)
        hsv = image.rgb_to_hsv(x)
        result = image.hsv_to_rgb(hsv)
        np.testing.assert_array_almost_equal(result.numpy(), x.numpy(), decimal=5)


class TestExtractPatches:
    """Test patch extraction operations."""
    
    def test_extract_patches(self):
        """Test patch extraction."""
        x = tf.random.uniform([1, 8, 8, 3])
        result = image.extract_patches(x, size=3, strides=1, padding="valid")
        # Output should have patches with flattened channels
        assert len(result.shape) == 4
    
    def test_extract_patches_with_stride(self):
        """Test patch extraction with stride."""
        x = tf.random.uniform([1, 8, 8, 3])
        result = image.extract_patches(x, size=2, strides=2, padding="valid")
        # With 8x8 input, 2x2 patches, stride 2: should get 4x4 output
        assert result.shape[1:3] == (4, 4)


class TestGaussianBlur:
    """Test Gaussian blur operations."""
    
    def test_gaussian_blur(self):
        """Test Gaussian blur."""
        x = tf.random.uniform([1, 16, 16, 3])
        result = image.gaussian_blur(x, kernel_size=3, sigma=1.0)
        assert result.shape == x.shape
    
    def test_gaussian_blur_large_kernel(self):
        """Test Gaussian blur with larger kernel."""
        x = tf.random.uniform([1, 16, 16, 3])
        result = image.gaussian_blur(x, kernel_size=5, sigma=2.0)
        assert result.shape == x.shape


class TestAffineTransform:
    """Test affine transformation operations."""
    
    def test_affine_transform_identity(self):
        """Test affine transform with identity matrix."""
        x = tf.random.uniform([1, 8, 8, 3])
        # Identity transformation (approximately - TF uses different parameterization)
        transform = tf.constant([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
        result = image.affine_transform(x, transform)
        assert result.shape == x.shape


class TestMapCoordinates:
    """Test map coordinates operations."""
    
    def test_map_coordinates_2d(self):
        """Test 2D map coordinates."""
        x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        coords = [
            tf.constant([0.0, 1.0]),  # y coordinates
            tf.constant([0.0, 1.0])   # x coordinates
        ]
        result = image.map_coordinates(x, coords, order=0)
        expected = np.array([1.0, 4.0])
        np.testing.assert_array_almost_equal(result.numpy(), expected)
    
    def test_map_coordinates_bilinear(self):
        """Test bilinear map coordinates."""
        x = tf.constant([[0.0, 1.0], [1.0, 2.0]])
        coords = [
            tf.constant([0.5]),  # y coordinates
            tf.constant([0.5])   # x coordinates
        ]
        result = image.map_coordinates(x, coords, order=1)
        # Bilinear interpolation at (0.5, 0.5) should be average of all 4 corners
        expected_value = (0.0 + 1.0 + 1.0 + 2.0) / 4
        np.testing.assert_almost_equal(result.numpy()[0], expected_value)
