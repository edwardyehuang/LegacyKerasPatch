"""Tests for core operations in legacy_keras_patch.ops."""

import pytest
import numpy as np
import tensorflow as tf

from legacy_keras_patch import ops


class TestCoreOps:
    """Test core operations."""
    
    def test_cast(self):
        """Test tensor casting."""
        x = tf.constant([1.0, 2.0, 3.0])
        result = ops.cast(x, tf.int32)
        assert result.dtype == tf.int32
        np.testing.assert_array_equal(result.numpy(), [1, 2, 3])
    
    def test_convert_to_tensor(self):
        """Test converting to tensor."""
        x = [1.0, 2.0, 3.0]
        result = ops.convert_to_tensor(x)
        assert tf.is_tensor(result)
        np.testing.assert_array_equal(result.numpy(), [1.0, 2.0, 3.0])
    
    def test_convert_to_tensor_with_dtype(self):
        """Test converting to tensor with dtype."""
        x = [1, 2, 3]
        result = ops.convert_to_tensor(x, dtype=tf.float32)
        assert result.dtype == tf.float32
    
    def test_convert_to_numpy(self):
        """Test converting to numpy."""
        x = tf.constant([1.0, 2.0, 3.0])
        result = ops.convert_to_numpy(x)
        assert isinstance(result, np.ndarray)
    
    def test_is_tensor(self):
        """Test is_tensor."""
        x = tf.constant([1.0, 2.0])
        assert ops.is_tensor(x)
        assert not ops.is_tensor([1.0, 2.0])
    
    def test_dtype(self):
        """Test dtype."""
        x = tf.constant([1.0, 2.0])
        assert ops.dtype(x) == tf.float32
    
    def test_shape(self):
        """Test shape."""
        x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        result = ops.shape(x)
        np.testing.assert_array_equal(result.numpy(), [2, 2])


class TestMathOps:
    """Test mathematical operations."""
    
    def test_erf(self):
        """Test error function."""
        x = tf.constant([0.0, 1.0])
        result = ops.erf(x)
        # erf(0) = 0
        np.testing.assert_almost_equal(result.numpy()[0], 0.0, decimal=5)
    
    def test_erfinv(self):
        """Test inverse error function."""
        x = tf.constant([0.0])
        result = ops.erfinv(x)
        # erfinv(0) = 0
        np.testing.assert_almost_equal(result.numpy()[0], 0.0, decimal=5)
    
    def test_rsqrt(self):
        """Test reciprocal square root."""
        x = tf.constant([4.0, 9.0, 16.0])
        result = ops.rsqrt(x)
        expected = np.array([0.5, 1/3, 0.25])
        np.testing.assert_array_almost_equal(result.numpy(), expected)
    
    def test_logsumexp(self):
        """Test log-sum-exp."""
        x = tf.constant([1.0, 2.0, 3.0])
        result = ops.logsumexp(x)
        expected = np.log(np.sum(np.exp([1.0, 2.0, 3.0])))
        np.testing.assert_almost_equal(result.numpy(), expected, decimal=5)
    
    def test_logdet(self):
        """Test log determinant."""
        # Create a positive definite matrix
        a = tf.constant([[4.0, 2.0], [2.0, 3.0]])
        matrix = tf.matmul(a, tf.transpose(a)) + tf.eye(2)
        result = ops.logdet(matrix)
        expected = np.log(np.linalg.det(matrix.numpy()))
        np.testing.assert_almost_equal(result.numpy(), expected, decimal=4)


class TestFFTOps:
    """Test FFT operations."""
    
    def test_fft(self):
        """Test 1D FFT."""
        x = tf.constant([1.0, 2.0, 3.0, 4.0], dtype=tf.float32)
        result = ops.fft(x)
        assert result.dtype == tf.complex64
        assert result.shape == x.shape
    
    def test_fft2(self):
        """Test 2D FFT."""
        x = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
        result = ops.fft2(x)
        assert result.dtype == tf.complex64
        assert result.shape == x.shape
    
    def test_rfft(self):
        """Test real FFT."""
        x = tf.constant([1.0, 2.0, 3.0, 4.0])
        result = ops.rfft(x)
        assert result.dtype == tf.complex64
    
    def test_irfft(self):
        """Test inverse real FFT."""
        x = tf.constant([1.0, 2.0, 3.0, 4.0])
        freq = ops.rfft(x)
        result = ops.irfft(freq)
        np.testing.assert_array_almost_equal(result.numpy(), x.numpy(), decimal=5)
    
    def test_ifft2(self):
        """Test 2D inverse FFT."""
        x = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
        freq = ops.fft2(x)
        result = ops.ifft2(freq)
        # Result should be close to original (may be complex with small imaginary parts)
        np.testing.assert_array_almost_equal(tf.math.real(result).numpy(), x.numpy(), decimal=5)


class TestControlFlow:
    """Test control flow operations."""
    
    def test_cond(self):
        """Test conditional execution."""
        x = tf.constant(5)
        result = ops.cond(
            x > 3,
            lambda: tf.constant(1),
            lambda: tf.constant(0)
        )
        assert result.numpy() == 1
    
    def test_while_loop(self):
        """Test while loop."""
        i = tf.constant(0)
        result = ops.while_loop(
            lambda i: i < 5,
            lambda i: (i + 1,),
            [i]
        )
        assert result[0].numpy() == 5
    
    def test_fori_loop(self):
        """Test fori loop."""
        init_val = tf.constant(0.0)
        result = ops.fori_loop(0, 5, lambda i, val: val + 1.0, init_val)
        assert result.numpy() == 5.0


class TestArrayOps:
    """Test array operations."""
    
    def test_slice(self):
        """Test slice operation."""
        x = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = ops.slice(x, [0, 1], [2, 2])
        expected = np.array([[2.0, 3.0], [5.0, 6.0]])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_slice_update(self):
        """Test slice update operation."""
        x = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        updates = tf.constant([[20.0, 30.0], [50.0, 60.0]])
        result = ops.slice_update(x, [0, 1], updates)
        expected = np.array([[1.0, 20.0, 30.0], [4.0, 50.0, 60.0]])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_slice_update_graph_mode(self):
        """Test slice update operation in TensorFlow graph mode."""
        x = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        updates = tf.constant([[20.0, 30.0], [50.0, 60.0]])

        @tf.function
        def update():
            return ops.slice_update(x, [0, 1], updates)

        expected = np.array([[1.0, 20.0, 30.0], [4.0, 50.0, 60.0]])
        np.testing.assert_array_equal(update().numpy(), expected)
    
    def test_scatter(self):
        """Test scatter operation."""
        indices = tf.constant([[0], [2]])
        values = tf.constant([1.0, 2.0])
        shape = [4]
        result = ops.scatter(indices, values, shape)
        expected = np.array([1.0, 0.0, 2.0, 0.0])
        np.testing.assert_array_equal(result.numpy(), expected)
    
    def test_scatter_update(self):
        """Test scatter update operation."""
        inputs = tf.constant([1.0, 2.0, 3.0, 4.0])
        indices = tf.constant([[1], [3]])
        updates = tf.constant([10.0, 20.0])
        result = ops.scatter_update(inputs, indices, updates)
        expected = np.array([1.0, 10.0, 3.0, 20.0])
        np.testing.assert_array_equal(result.numpy(), expected)
    
    def test_unstack(self):
        """Test unstack operation."""
        x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        result = ops.unstack(x, axis=0)
        assert len(result) == 2
        np.testing.assert_array_equal(result[0].numpy(), [1.0, 2.0])
    
    def test_top_k(self):
        """Test top k operation."""
        x = tf.constant([1.0, 5.0, 3.0, 4.0, 2.0])
        values, indices = ops.top_k(x, k=3)
        np.testing.assert_array_equal(values.numpy(), [5.0, 4.0, 3.0])
        np.testing.assert_array_equal(indices.numpy(), [1, 3, 2])


class TestSegmentOps:
    """Test segment operations."""
    
    def test_segment_sum_sorted(self):
        """Test sorted segment sum."""
        data = tf.constant([1.0, 2.0, 3.0, 4.0])
        segment_ids = tf.constant([0, 0, 1, 1])
        result = ops.segment_sum(data, segment_ids, sorted=True)
        expected = np.array([3.0, 7.0])
        np.testing.assert_array_equal(result.numpy(), expected)
    
    def test_segment_max_sorted(self):
        """Test sorted segment max."""
        data = tf.constant([1.0, 4.0, 2.0, 3.0])
        segment_ids = tf.constant([0, 0, 1, 1])
        result = ops.segment_max(data, segment_ids, sorted=True)
        expected = np.array([4.0, 3.0])
        np.testing.assert_array_equal(result.numpy(), expected)


class TestComplexOps:
    """Test complex number operations."""
    
    def test_view_as_complex(self):
        """Test viewing real tensor as complex."""
        x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        result = ops.view_as_complex(x)
        assert result.dtype == tf.complex64
        # First element should be 1+2j
        np.testing.assert_almost_equal(result.numpy()[0].real, 1.0)
        np.testing.assert_almost_equal(result.numpy()[0].imag, 2.0)
    
    def test_view_as_real(self):
        """Test viewing complex tensor as real."""
        x = tf.constant([1+2j, 3+4j], dtype=tf.complex64)
        result = ops.view_as_real(x)
        expected = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_almost_equal(result.numpy(), expected)
    
    def test_view_as_complex_roundtrip(self):
        """Test complex-real roundtrip."""
        x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        complex_x = ops.view_as_complex(x)
        result = ops.view_as_real(complex_x)
        np.testing.assert_array_almost_equal(result.numpy(), x.numpy())


class TestGradientOps:
    """Test gradient operations."""
    
    def test_stop_gradient(self):
        """Test stop gradient."""
        x = tf.constant([1.0, 2.0, 3.0])
        result = ops.stop_gradient(x)
        np.testing.assert_array_equal(result.numpy(), x.numpy())
        
        # Verify gradient stops
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = ops.stop_gradient(x) * 2
        grad = tape.gradient(y, x)
        assert grad is None
    
    def test_custom_gradient(self):
        """Test custom gradient decorator."""
        @ops.custom_gradient
        def my_func(x):
            y = x * 2
            def grad(dy):
                return dy * 3  # Custom gradient: return 3 instead of 2
            return y, grad
        
        x = tf.constant(1.0)
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = my_func(x)
        grad = tape.gradient(y, x)
        assert grad.numpy() == 3.0


class TestMapOps:
    """Test map operations."""
    
    def test_map(self):
        """Test map function."""
        xs = tf.constant([1.0, 2.0, 3.0])
        result = ops.map(lambda x: x * 2, xs)
        expected = np.array([2.0, 4.0, 6.0])
        np.testing.assert_array_equal(result.numpy(), expected)
    
    def test_vectorized_map(self):
        """Test vectorized map."""
        xs = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        result = ops.vectorized_map(lambda x: x * 2, xs)
        expected = np.array([[2.0, 4.0], [6.0, 8.0]])
        np.testing.assert_array_equal(result.numpy(), expected)


class TestExtractSequences:
    """Test sequence extraction."""
    
    def test_extract_sequences(self):
        """Test extract sequences."""
        x = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ops.extract_sequences(x, sequence_length=3, sequence_stride=1)
        # Should get sliding windows
        assert result.shape[0] == 3  # 3 sequences of length 3


class TestNormOps:
    """Test norm operations at ops level."""
    
    def test_norm_vector(self):
        """Test vector norm."""
        x = tf.constant([3.0, 4.0])
        result = ops.norm(x)
        np.testing.assert_almost_equal(result.numpy(), 5.0)
    
    def test_norm_matrix(self):
        """Test matrix norm."""
        x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        result = ops.norm(x, ord='fro')
        expected = np.sqrt(1 + 4 + 9 + 16)
        np.testing.assert_almost_equal(result.numpy(), expected, decimal=5)


class TestSubmodules:
    """Test that submodules are accessible from ops."""
    
    def test_nn_accessible(self):
        """Test nn submodule is accessible."""
        assert hasattr(ops, 'nn')
        assert hasattr(ops.nn, 'relu')
    
    def test_image_accessible(self):
        """Test image submodule is accessible."""
        assert hasattr(ops, 'image')
        assert hasattr(ops.image, 'resize')
    
    def test_linalg_accessible(self):
        """Test linalg submodule is accessible."""
        assert hasattr(ops, 'linalg')
        assert hasattr(ops.linalg, 'det')
    
    def test_numpy_accessible(self):
        """Test numpy submodule is accessible."""
        assert hasattr(ops, 'numpy')
        assert hasattr(ops.numpy, 'abs')


class TestReexports:
    """Test that operations are re-exported at ops level."""
    
    def test_numpy_ops_at_top_level(self):
        """Test numpy ops are available at ops level."""
        # These should be re-exported from numpy_ops
        assert hasattr(ops, 'abs')
        assert hasattr(ops, 'sum')
        assert hasattr(ops, 'mean')
        assert hasattr(ops, 'zeros')
        assert hasattr(ops, 'ones')
        assert hasattr(ops, 'matmul')
    
    def test_nn_ops_at_top_level(self):
        """Test nn ops are available at ops level."""
        # These should be re-exported from nn
        assert hasattr(ops, 'relu')
        assert hasattr(ops, 'sigmoid')
        assert hasattr(ops, 'softmax')
        assert hasattr(ops, 'gelu')
    
    def test_linalg_ops_at_top_level(self):
        """Test linalg ops are available at ops level."""
        # These should be re-exported from linalg
        assert hasattr(ops, 'cholesky')
        assert hasattr(ops, 'det')
        assert hasattr(ops, 'inv')
        assert hasattr(ops, 'svd')


class TestSaturateCast:
    """Test saturate_cast operation."""
    
    def test_saturate_cast(self):
        """Test saturate cast."""
        x = tf.constant([0.0, 128.0, 300.0])
        result = ops.saturate_cast(x, tf.uint8)
        # Values should be clipped to uint8 range [0, 255]
        expected = np.array([0, 128, 255], dtype=np.uint8)
        np.testing.assert_array_equal(result.numpy(), expected)
