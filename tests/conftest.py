"""Pytest configuration and fixtures for LegacyKerasPatch tests."""

import pytest
import numpy as np
import tensorflow as tf


@pytest.fixture
def sample_tensor():
    """Create a sample tensor for testing."""
    return tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


@pytest.fixture
def sample_image():
    """Create a sample image tensor for testing (NHWC format)."""
    return tf.random.uniform([2, 8, 8, 3], minval=0, maxval=1)


@pytest.fixture
def sample_matrix():
    """Create a sample square matrix for testing."""
    return tf.constant([[4.0, 2.0], [2.0, 3.0]])


@pytest.fixture
def positive_definite_matrix():
    """Create a positive definite matrix for testing."""
    # Create a symmetric positive definite matrix
    a = tf.constant([[2.0, 1.0], [1.0, 3.0]])
    return tf.matmul(a, tf.transpose(a)) + tf.eye(2)
