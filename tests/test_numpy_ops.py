"""Tests for numpy operations in legacy_keras_patch.ops."""

import pytest
import numpy as np
import tensorflow as tf

from legacy_keras_patch.ops import numpy_ops


class TestBasicMath:
    """Test basic math operations."""
    
    def test_abs(self):
        """Test absolute value."""
        x = tf.constant([-1.0, 2.0, -3.0])
        result = numpy_ops.abs(x)
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(result.numpy(), expected)
    
    def test_absolute(self):
        """Test absolute (alias for abs)."""
        x = tf.constant([-1.0, 2.0, -3.0])
        result = numpy_ops.absolute(x)
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(result.numpy(), expected)
    
    def test_add(self):
        """Test addition."""
        x = tf.constant([1.0, 2.0, 3.0])
        y = tf.constant([4.0, 5.0, 6.0])
        result = numpy_ops.add(x, y)
        expected = np.array([5.0, 7.0, 9.0])
        np.testing.assert_array_almost_equal(result.numpy(), expected)
    
    def test_subtract(self):
        """Test subtraction."""
        x = tf.constant([5.0, 6.0, 7.0])
        y = tf.constant([1.0, 2.0, 3.0])
        result = numpy_ops.subtract(x, y)
        expected = np.array([4.0, 4.0, 4.0])
        np.testing.assert_array_almost_equal(result.numpy(), expected)
    
    def test_multiply(self):
        """Test multiplication."""
        x = tf.constant([1.0, 2.0, 3.0])
        y = tf.constant([4.0, 5.0, 6.0])
        result = numpy_ops.multiply(x, y)
        expected = np.array([4.0, 10.0, 18.0])
        np.testing.assert_array_almost_equal(result.numpy(), expected)
    
    def test_divide(self):
        """Test division."""
        x = tf.constant([4.0, 10.0, 18.0])
        y = tf.constant([2.0, 5.0, 6.0])
        result = numpy_ops.divide(x, y)
        expected = np.array([2.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(result.numpy(), expected)
    
    def test_sqrt(self):
        """Test square root."""
        x = tf.constant([1.0, 4.0, 9.0])
        result = numpy_ops.sqrt(x)
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(result.numpy(), expected)
    
    def test_square(self):
        """Test square."""
        x = tf.constant([1.0, 2.0, 3.0])
        result = numpy_ops.square(x)
        expected = np.array([1.0, 4.0, 9.0])
        np.testing.assert_array_almost_equal(result.numpy(), expected)
    
    def test_power(self):
        """Test power."""
        x = tf.constant([2.0, 3.0, 4.0])
        result = numpy_ops.power(x, 2)
        expected = np.array([4.0, 9.0, 16.0])
        np.testing.assert_array_almost_equal(result.numpy(), expected)
    
    def test_exp(self):
        """Test exponential."""
        x = tf.constant([0.0, 1.0, 2.0])
        result = numpy_ops.exp(x)
        expected = np.exp(np.array([0.0, 1.0, 2.0]))
        np.testing.assert_array_almost_equal(result.numpy(), expected)
    
    def test_log(self):
        """Test natural logarithm."""
        x = tf.constant([1.0, np.e, np.e**2])
        result = numpy_ops.log(x)
        expected = np.array([0.0, 1.0, 2.0])
        np.testing.assert_array_almost_equal(result.numpy(), expected, decimal=5)


class TestTrigonometric:
    """Test trigonometric operations."""
    
    def test_sin(self):
        """Test sine."""
        x = tf.constant([0.0, np.pi/2, np.pi])
        result = numpy_ops.sin(x)
        expected = np.sin(np.array([0.0, np.pi/2, np.pi]))
        np.testing.assert_array_almost_equal(result.numpy(), expected)
    
    def test_cos(self):
        """Test cosine."""
        x = tf.constant([0.0, np.pi/2, np.pi])
        result = numpy_ops.cos(x)
        expected = np.cos(np.array([0.0, np.pi/2, np.pi]))
        np.testing.assert_array_almost_equal(result.numpy(), expected)
    
    def test_tan(self):
        """Test tangent."""
        x = tf.constant([0.0, np.pi/4])
        result = numpy_ops.tan(x)
        expected = np.tan(np.array([0.0, np.pi/4]))
        np.testing.assert_array_almost_equal(result.numpy(), expected)
    
    def test_arcsin(self):
        """Test inverse sine."""
        x = tf.constant([0.0, 0.5, 1.0])
        result = numpy_ops.arcsin(x)
        expected = np.arcsin(np.array([0.0, 0.5, 1.0]))
        np.testing.assert_array_almost_equal(result.numpy(), expected)
    
    def test_arccos(self):
        """Test inverse cosine."""
        x = tf.constant([0.0, 0.5, 1.0])
        result = numpy_ops.arccos(x)
        expected = np.arccos(np.array([0.0, 0.5, 1.0]))
        np.testing.assert_array_almost_equal(result.numpy(), expected)
    
    def test_arctan(self):
        """Test inverse tangent."""
        x = tf.constant([0.0, 1.0])
        result = numpy_ops.arctan(x)
        expected = np.arctan(np.array([0.0, 1.0]))
        np.testing.assert_array_almost_equal(result.numpy(), expected)


class TestReductions:
    """Test reduction operations."""
    
    def test_sum(self):
        """Test sum."""
        x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        result = numpy_ops.sum(x)
        assert result.numpy() == 10.0
    
    def test_sum_axis(self):
        """Test sum along axis."""
        x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        result = numpy_ops.sum(x, axis=0)
        expected = np.array([4.0, 6.0])
        np.testing.assert_array_almost_equal(result.numpy(), expected)
    
    def test_mean(self):
        """Test mean."""
        x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        result = numpy_ops.mean(x)
        assert result.numpy() == 2.5
    
    def test_max(self):
        """Test max."""
        x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        result = numpy_ops.max(x)
        assert result.numpy() == 4.0
    
    def test_min(self):
        """Test min."""
        x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        result = numpy_ops.min(x)
        assert result.numpy() == 1.0
    
    def test_prod(self):
        """Test prod."""
        x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        result = numpy_ops.prod(x)
        assert result.numpy() == 24.0
    
    def test_std(self):
        """Test standard deviation."""
        x = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])
        result = numpy_ops.std(x)
        expected = np.std(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        np.testing.assert_almost_equal(result.numpy(), expected, decimal=5)
    
    def test_var(self):
        """Test variance."""
        x = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])
        result = numpy_ops.var(x)
        expected = np.var(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        np.testing.assert_almost_equal(result.numpy(), expected, decimal=5)


class TestArrayCreation:
    """Test array creation operations."""
    
    def test_zeros(self):
        """Test zeros."""
        result = numpy_ops.zeros((2, 3))
        expected = np.zeros((2, 3))
        np.testing.assert_array_equal(result.numpy(), expected)
    
    def test_ones(self):
        """Test ones."""
        result = numpy_ops.ones((2, 3))
        expected = np.ones((2, 3))
        np.testing.assert_array_equal(result.numpy(), expected)
    
    def test_full(self):
        """Test full."""
        result = numpy_ops.full((2, 3), 5.0)
        expected = np.full((2, 3), 5.0)
        np.testing.assert_array_equal(result.numpy(), expected)
    
    def test_eye(self):
        """Test eye (identity matrix)."""
        result = numpy_ops.eye(3)
        expected = np.eye(3)
        np.testing.assert_array_equal(result.numpy(), expected)
    
    def test_arange(self):
        """Test arange."""
        result = numpy_ops.arange(0, 5, 1)
        expected = np.arange(0, 5, 1)
        np.testing.assert_array_equal(result.numpy(), expected)
    
    def test_linspace(self):
        """Test linspace."""
        result = numpy_ops.linspace(0.0, 1.0, 5)
        expected = np.linspace(0.0, 1.0, 5)
        np.testing.assert_array_almost_equal(result.numpy(), expected)


class TestArrayManipulation:
    """Test array manipulation operations."""
    
    def test_reshape(self):
        """Test reshape."""
        x = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result = numpy_ops.reshape(x, (2, 3))
        assert result.shape == (2, 3)
    
    def test_transpose(self):
        """Test transpose."""
        x = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = numpy_ops.transpose(x)
        expected = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
        np.testing.assert_array_equal(result.numpy(), expected)
    
    def test_concatenate(self):
        """Test concatenate."""
        x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        y = tf.constant([[5.0, 6.0], [7.0, 8.0]])
        result = numpy_ops.concatenate([x, y], axis=0)
        assert result.shape == (4, 2)
    
    def test_stack(self):
        """Test stack."""
        x = tf.constant([1.0, 2.0, 3.0])
        y = tf.constant([4.0, 5.0, 6.0])
        result = numpy_ops.stack([x, y], axis=0)
        assert result.shape == (2, 3)
    
    def test_squeeze(self):
        """Test squeeze."""
        x = tf.constant([[[1.0, 2.0, 3.0]]])
        result = numpy_ops.squeeze(x)
        assert result.shape == (3,)
    
    def test_expand_dims(self):
        """Test expand_dims."""
        x = tf.constant([1.0, 2.0, 3.0])
        result = numpy_ops.expand_dims(x, axis=0)
        assert result.shape == (1, 3)
    
    def test_flip(self):
        """Test flip."""
        x = tf.constant([1.0, 2.0, 3.0])
        result = numpy_ops.flip(x)
        expected = np.array([3.0, 2.0, 1.0])
        np.testing.assert_array_equal(result.numpy(), expected)


class TestComparison:
    """Test comparison operations."""
    
    def test_equal(self):
        """Test equal."""
        x = tf.constant([1.0, 2.0, 3.0])
        y = tf.constant([1.0, 0.0, 3.0])
        result = numpy_ops.equal(x, y)
        expected = np.array([True, False, True])
        np.testing.assert_array_equal(result.numpy(), expected)
    
    def test_not_equal(self):
        """Test not_equal."""
        x = tf.constant([1.0, 2.0, 3.0])
        y = tf.constant([1.0, 0.0, 3.0])
        result = numpy_ops.not_equal(x, y)
        expected = np.array([False, True, False])
        np.testing.assert_array_equal(result.numpy(), expected)
    
    def test_greater(self):
        """Test greater."""
        x = tf.constant([1.0, 2.0, 3.0])
        y = tf.constant([0.0, 2.0, 4.0])
        result = numpy_ops.greater(x, y)
        expected = np.array([True, False, False])
        np.testing.assert_array_equal(result.numpy(), expected)
    
    def test_less(self):
        """Test less."""
        x = tf.constant([1.0, 2.0, 3.0])
        y = tf.constant([0.0, 2.0, 4.0])
        result = numpy_ops.less(x, y)
        expected = np.array([False, False, True])
        np.testing.assert_array_equal(result.numpy(), expected)


class TestLogical:
    """Test logical operations."""
    
    def test_all(self):
        """Test all."""
        x = tf.constant([True, True, True])
        result = numpy_ops.all(x)
        assert result.numpy() == True
        
        y = tf.constant([True, False, True])
        result = numpy_ops.all(y)
        assert result.numpy() == False
    
    def test_any(self):
        """Test any."""
        x = tf.constant([False, False, False])
        result = numpy_ops.any(x)
        assert result.numpy() == False
        
        y = tf.constant([True, False, False])
        result = numpy_ops.any(y)
        assert result.numpy() == True
    
    def test_logical_and(self):
        """Test logical_and."""
        x = tf.constant([True, True, False, False])
        y = tf.constant([True, False, True, False])
        result = numpy_ops.logical_and(x, y)
        expected = np.array([True, False, False, False])
        np.testing.assert_array_equal(result.numpy(), expected)
    
    def test_logical_or(self):
        """Test logical_or."""
        x = tf.constant([True, True, False, False])
        y = tf.constant([True, False, True, False])
        result = numpy_ops.logical_or(x, y)
        expected = np.array([True, True, True, False])
        np.testing.assert_array_equal(result.numpy(), expected)
    
    def test_logical_not(self):
        """Test logical_not."""
        x = tf.constant([True, False])
        result = numpy_ops.logical_not(x)
        expected = np.array([False, True])
        np.testing.assert_array_equal(result.numpy(), expected)


class TestMatrixOps:
    """Test matrix operations."""
    
    def test_matmul(self):
        """Test matrix multiplication."""
        x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        y = tf.constant([[5.0, 6.0], [7.0, 8.0]])
        result = numpy_ops.matmul(x, y)
        expected = np.matmul(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[5.0, 6.0], [7.0, 8.0]]))
        np.testing.assert_array_almost_equal(result.numpy(), expected)
    
    def test_dot(self):
        """Test dot product."""
        x = tf.constant([1.0, 2.0, 3.0])
        y = tf.constant([4.0, 5.0, 6.0])
        result = numpy_ops.dot(x, y)
        expected = np.dot(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))
        np.testing.assert_almost_equal(result.numpy(), expected)


class TestMisc:
    """Test miscellaneous operations."""
    
    def test_clip(self):
        """Test clip."""
        x = tf.constant([1.0, 5.0, 10.0, 15.0])
        result = numpy_ops.clip(x, 3.0, 12.0)
        expected = np.array([3.0, 5.0, 10.0, 12.0])
        np.testing.assert_array_almost_equal(result.numpy(), expected)
    
    def test_where(self):
        """Test where."""
        condition = tf.constant([True, False, True])
        x = tf.constant([1.0, 2.0, 3.0])
        y = tf.constant([4.0, 5.0, 6.0])
        result = numpy_ops.where(condition, x, y)
        expected = np.array([1.0, 5.0, 3.0])
        np.testing.assert_array_almost_equal(result.numpy(), expected)
    
    def test_argmax(self):
        """Test argmax."""
        x = tf.constant([1.0, 3.0, 2.0])
        result = numpy_ops.argmax(x)
        assert result.numpy() == 1
    
    def test_argmin(self):
        """Test argmin."""
        x = tf.constant([3.0, 1.0, 2.0])
        result = numpy_ops.argmin(x)
        assert result.numpy() == 1
    
    def test_sort(self):
        """Test sort."""
        x = tf.constant([3.0, 1.0, 2.0])
        result = numpy_ops.sort(x)
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(result.numpy(), expected)
