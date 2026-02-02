"""Tests for neural network operations in legacy_keras_patch.ops.nn."""

import pytest
import numpy as np
import tensorflow as tf

from legacy_keras_patch.ops import nn


class TestActivations:
    """Test activation functions."""
    
    def test_relu(self):
        """Test ReLU activation."""
        x = tf.constant([-1.0, 0.0, 1.0, 2.0])
        result = nn.relu(x)
        expected = np.array([0.0, 0.0, 1.0, 2.0])
        np.testing.assert_array_almost_equal(result.numpy(), expected)
    
    def test_relu6(self):
        """Test ReLU6 activation."""
        x = tf.constant([-1.0, 0.0, 3.0, 7.0])
        result = nn.relu6(x)
        expected = np.array([0.0, 0.0, 3.0, 6.0])
        np.testing.assert_array_almost_equal(result.numpy(), expected)
    
    def test_sigmoid(self):
        """Test sigmoid activation."""
        x = tf.constant([0.0])
        result = nn.sigmoid(x)
        assert abs(result.numpy()[0] - 0.5) < 1e-6
    
    def test_softmax(self):
        """Test softmax activation."""
        x = tf.constant([[1.0, 2.0, 3.0]])
        result = nn.softmax(x)
        # Sum should be 1
        np.testing.assert_almost_equal(np.sum(result.numpy()), 1.0)
    
    def test_softplus(self):
        """Test softplus activation."""
        x = tf.constant([0.0, 1.0])
        result = nn.softplus(x)
        expected = np.log(1 + np.exp(np.array([0.0, 1.0])))
        np.testing.assert_array_almost_equal(result.numpy(), expected)
    
    def test_softsign(self):
        """Test softsign activation."""
        x = tf.constant([0.0, 1.0, -1.0])
        result = nn.softsign(x)
        expected = np.array([0.0, 0.5, -0.5])
        np.testing.assert_array_almost_equal(result.numpy(), expected)
    
    def test_selu(self):
        """Test SELU activation."""
        x = tf.constant([0.0, 1.0])
        result = nn.selu(x)
        # Just verify it runs and has reasonable values
        assert result.shape == x.shape
    
    def test_elu(self):
        """Test ELU activation."""
        x = tf.constant([-1.0, 0.0, 1.0])
        result = nn.elu(x)
        # Positive values should be unchanged
        assert result.numpy()[2] == 1.0
        # Negative values should be transformed
        assert result.numpy()[0] < 0
    
    def test_leaky_relu(self):
        """Test leaky ReLU activation."""
        x = tf.constant([-1.0, 0.0, 1.0])
        result = nn.leaky_relu(x, negative_slope=0.1)
        expected = np.array([-0.1, 0.0, 1.0])
        np.testing.assert_array_almost_equal(result.numpy(), expected)
    
    def test_gelu(self):
        """Test GELU activation."""
        x = tf.constant([0.0, 1.0])
        result = nn.gelu(x)
        # Just verify it runs
        assert result.shape == x.shape
    
    def test_silu(self):
        """Test SiLU (Swish) activation."""
        x = tf.constant([0.0])
        result = nn.silu(x)
        assert result.numpy()[0] == 0.0
    
    def test_swish(self):
        """Test Swish activation (alias for silu)."""
        x = tf.constant([0.0])
        result = nn.swish(x)
        assert result.numpy()[0] == 0.0
    
    def test_hard_sigmoid(self):
        """Test hard sigmoid activation."""
        x = tf.constant([-3.0, 0.0, 3.0])
        result = nn.hard_sigmoid(x)
        # Values should be clipped between 0 and 1
        assert all(result.numpy() >= 0)
        assert all(result.numpy() <= 1)
    
    def test_hard_silu(self):
        """Test hard SiLU activation."""
        x = tf.constant([0.0, 1.0])
        result = nn.hard_silu(x)
        assert result.shape == x.shape
    
    def test_hard_tanh(self):
        """Test hard tanh activation."""
        x = tf.constant([-2.0, 0.0, 2.0])
        result = nn.hard_tanh(x)
        expected = np.array([-1.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(result.numpy(), expected)
    
    def test_log_sigmoid(self):
        """Test log sigmoid activation."""
        x = tf.constant([0.0])
        result = nn.log_sigmoid(x)
        expected = np.log(0.5)
        np.testing.assert_almost_equal(result.numpy()[0], expected, decimal=5)
    
    def test_log_softmax(self):
        """Test log softmax activation."""
        x = tf.constant([[1.0, 2.0, 3.0]])
        result = nn.log_softmax(x)
        # exp of log softmax should sum to 1
        exp_sum = np.sum(np.exp(result.numpy()))
        np.testing.assert_almost_equal(exp_sum, 1.0, decimal=5)


class TestNormalization:
    """Test normalization operations."""
    
    def test_batch_normalization(self):
        """Test batch normalization."""
        x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        mean = tf.constant([2.0, 3.0])
        variance = tf.constant([1.0, 1.0])
        result = nn.batch_normalization(x, mean, variance, axis=-1)
        # Just verify it runs
        assert result.shape == x.shape
    
    def test_layer_normalization(self):
        """Test layer normalization."""
        x = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = nn.layer_normalization(x)
        # Mean along last axis should be close to 0
        np.testing.assert_almost_equal(np.mean(result.numpy(), axis=-1), [0.0, 0.0], decimal=5)
    
    def test_rms_normalization(self):
        """Test RMS normalization."""
        x = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = nn.rms_normalization(x)
        assert result.shape == x.shape


class TestPooling:
    """Test pooling operations."""
    
    def test_max_pool_2d(self):
        """Test 2D max pooling."""
        x = tf.constant([[[[1.0], [2.0]], [[3.0], [4.0]]]])  # (1, 2, 2, 1)
        result = nn.max_pool(x, pool_size=2, strides=2)
        assert result.numpy()[0, 0, 0, 0] == 4.0
    
    def test_average_pool_2d(self):
        """Test 2D average pooling."""
        x = tf.constant([[[[1.0], [2.0]], [[3.0], [4.0]]]])  # (1, 2, 2, 1)
        result = nn.average_pool(x, pool_size=2, strides=2)
        assert result.numpy()[0, 0, 0, 0] == 2.5
    
    def test_adaptive_average_pool(self):
        """Test adaptive average pooling."""
        x = tf.random.uniform([1, 8, 8, 3])
        result = nn.adaptive_average_pool(x, output_size=(2, 2))
        assert result.shape[1:3] == (2, 2)
    
    def test_adaptive_max_pool(self):
        """Test adaptive max pooling."""
        x = tf.random.uniform([1, 8, 8, 3])
        result = nn.adaptive_max_pool(x, output_size=(2, 2))
        assert result.shape[1:3] == (2, 2)


class TestLoss:
    """Test loss functions."""
    
    def test_binary_crossentropy(self):
        """Test binary crossentropy loss."""
        target = tf.constant([1.0, 0.0])
        output = tf.constant([0.9, 0.1])
        result = nn.binary_crossentropy(target, output)
        assert result.shape == target.shape
        assert all(result.numpy() >= 0)
    
    def test_binary_crossentropy_from_logits(self):
        """Test binary crossentropy from logits."""
        target = tf.constant([1.0, 0.0])
        output = tf.constant([2.0, -2.0])
        result = nn.binary_crossentropy(target, output, from_logits=True)
        assert result.shape == target.shape
    
    def test_categorical_crossentropy(self):
        """Test categorical crossentropy loss."""
        target = tf.constant([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        output = tf.constant([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]])
        result = nn.categorical_crossentropy(target, output)
        assert len(result.shape) == 1
        assert all(result.numpy() >= 0)
    
    def test_sparse_categorical_crossentropy(self):
        """Test sparse categorical crossentropy loss."""
        target = tf.constant([0, 1])
        output = tf.constant([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]])
        result = nn.sparse_categorical_crossentropy(target, output)
        assert all(result.numpy() >= 0)


class TestConvolution:
    """Test convolution operations."""
    
    def test_conv2d(self):
        """Test 2D convolution."""
        x = tf.random.uniform([1, 8, 8, 3])
        kernel = tf.random.uniform([3, 3, 3, 16])
        result = nn.conv(x, kernel, strides=1, padding="valid")
        assert result.shape[-1] == 16
    
    def test_depthwise_conv2d(self):
        """Test 2D depthwise convolution."""
        x = tf.random.uniform([1, 8, 8, 3])
        kernel = tf.random.uniform([3, 3, 3, 1])
        result = nn.depthwise_conv(x, kernel, strides=1, padding="valid")
        assert result.shape[-1] == 3  # Same as input channels


class TestEncoding:
    """Test encoding operations."""
    
    def test_one_hot(self):
        """Test one-hot encoding."""
        x = tf.constant([0, 1, 2])
        result = nn.one_hot(x, num_classes=3)
        expected = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        np.testing.assert_array_almost_equal(result.numpy(), expected)
    
    def test_multi_hot(self):
        """Test multi-hot encoding."""
        x = tf.constant([[0, 1], [1, 2]])
        result = nn.multi_hot(x, num_classes=3)
        assert result.shape[-1] == 3


class TestAttention:
    """Test attention operations."""
    
    def test_dot_product_attention(self):
        """Test scaled dot product attention."""
        batch_size = 2
        seq_len = 4
        d_model = 8
        
        query = tf.random.uniform([batch_size, seq_len, d_model])
        key = tf.random.uniform([batch_size, seq_len, d_model])
        value = tf.random.uniform([batch_size, seq_len, d_model])
        
        result = nn.dot_product_attention(query, key, value)
        assert result.shape == (batch_size, seq_len, d_model)


class TestMisc:
    """Test miscellaneous nn operations."""
    
    def test_normalize(self):
        """Test L2 normalize."""
        x = tf.constant([1.0, 2.0, 3.0])
        result = nn.normalize(x)
        # L2 norm should be 1
        l2_norm = np.sqrt(np.sum(result.numpy() ** 2))
        np.testing.assert_almost_equal(l2_norm, 1.0)
    
    def test_moments(self):
        """Test moments (mean and variance)."""
        x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        mean, variance = nn.moments(x, axes=[0, 1])
        np.testing.assert_almost_equal(mean.numpy(), 2.5)
    
    def test_psnr(self):
        """Test PSNR."""
        x1 = tf.constant([[[[1.0]], [[2.0]]]])
        x2 = tf.constant([[[[1.0]], [[2.0]]]])
        result = nn.psnr(x1, x2, max_val=255.0)
        # Identical images should have infinite PSNR (or very high value)
        assert result.numpy() > 50  # Very high PSNR for identical images


class TestActivationEdgeCases:
    """Test edge cases for activation functions."""
    
    def test_celu(self):
        """Test CELU activation."""
        x = tf.constant([-1.0, 0.0, 1.0])
        result = nn.celu(x, alpha=1.0)
        assert result.shape == x.shape
    
    def test_glu(self):
        """Test GLU activation."""
        x = tf.constant([[1.0, 2.0, 3.0, 4.0]])
        result = nn.glu(x, axis=-1)
        assert result.shape[-1] == 2
    
    def test_hard_shrink(self):
        """Test hard shrink activation."""
        x = tf.constant([-1.0, -0.3, 0.0, 0.3, 1.0])
        result = nn.hard_shrink(x, threshold=0.5)
        expected = np.array([-1.0, 0.0, 0.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(result.numpy(), expected)
    
    def test_soft_shrink(self):
        """Test soft shrink activation."""
        x = tf.constant([-2.0, -0.3, 0.0, 0.3, 2.0])
        result = nn.soft_shrink(x, threshold=0.5)
        # Values within threshold should be 0
        assert result.numpy()[1] == 0.0
        assert result.numpy()[3] == 0.0
    
    def test_tanh_shrink(self):
        """Test tanh shrink activation."""
        x = tf.constant([0.0, 1.0])
        result = nn.tanh_shrink(x)
        expected = x.numpy() - np.tanh(x.numpy())
        np.testing.assert_array_almost_equal(result.numpy(), expected)
    
    def test_threshold(self):
        """Test threshold activation."""
        x = tf.constant([0.5, 1.5, 2.5])
        result = nn.threshold(x, threshold_value=1.0, value=0.0)
        expected = np.array([0.0, 1.5, 2.5])
        np.testing.assert_array_almost_equal(result.numpy(), expected)
    
    def test_squareplus(self):
        """Test squareplus activation."""
        x = tf.constant([0.0, 1.0])
        result = nn.squareplus(x)
        # squareplus(0) should be sqrt(b)/2 = 1 for b=4
        assert result.shape == x.shape
    
    def test_sparse_plus(self):
        """Test sparse plus activation."""
        x = tf.constant([-2.0, 0.0, 2.0])
        result = nn.sparse_plus(x)
        assert result.shape == x.shape
    
    def test_sparse_sigmoid(self):
        """Test sparse sigmoid activation."""
        x = tf.constant([-1.0, 0.0, 1.0])
        result = nn.sparse_sigmoid(x)
        # Should be clipped between 0 and 1
        assert all(result.numpy() >= 0)
        assert all(result.numpy() <= 1)
    
    def test_polar(self):
        """Test polar activation."""
        x = tf.constant([0.0, 2.0])
        result = nn.polar(x)
        expected = np.tanh(x.numpy() / 2)
        np.testing.assert_array_almost_equal(result.numpy(), expected)
