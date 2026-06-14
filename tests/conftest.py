"""Pytest configuration and fixtures for LegacyKerasPatch tests."""

import pytest
import numpy as np
import tensorflow as tf

# Configure multiple virtual CPUs for dtensor distribution tests.
# This must happen before any TF operations use the devices.
_physical_cpus = tf.config.list_physical_devices("CPU")
if _physical_cpus:
    try:
        tf.config.set_logical_device_configuration(
            _physical_cpus[0],
            [tf.config.LogicalDeviceConfiguration() for _ in range(8)],
        )
    except RuntimeError:
        # Devices already configured (e.g., re-import)
        pass


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
