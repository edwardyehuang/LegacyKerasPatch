"""Tests for random operations in legacy_keras_patch.random."""

import pytest
import numpy as np
import tensorflow as tf

from legacy_keras_patch import random


class TestSeedGenerator:
    """Test SeedGenerator class."""

    def test_create_with_seed(self):
        """Test creating SeedGenerator with explicit seed."""
        gen = random.SeedGenerator(42)
        assert gen._seed == 42

    def test_create_without_seed(self):
        """Test creating SeedGenerator without seed."""
        gen = random.SeedGenerator()
        assert isinstance(gen._seed, int)

    def test_next(self):
        """Test SeedGenerator.next() returns different seeds."""
        gen = random.SeedGenerator(42)
        s1 = gen.next()
        s2 = gen.next()
        assert s1 != s2

    def test_next_count(self):
        """Test SeedGenerator.next(count=N) returns N seeds."""
        gen = random.SeedGenerator(42)
        seeds = gen.next(count=3)
        assert len(seeds) == 3
        assert len(set(seeds)) == 3  # All different

    def test_from_seed_generator(self):
        """Test creating SeedGenerator from another SeedGenerator."""
        gen1 = random.SeedGenerator(42)
        gen2 = random.SeedGenerator(gen1)
        assert gen2._seed == 42


class TestNormal:
    """Test normal distribution sampling."""

    def test_shape(self):
        """Test output shape."""
        result = random.normal((3, 4))
        assert result.shape == (3, 4)

    def test_dtype(self):
        """Test output dtype."""
        result = random.normal((3,), dtype="float32")
        assert result.dtype == tf.float32

    def test_dtype_float64(self):
        """Test output dtype float64."""
        result = random.normal((3,), dtype="float64")
        assert result.dtype == tf.float64

    def test_mean_stddev(self):
        """Test mean and stddev parameters."""
        result = random.normal((10000,), mean=5.0, stddev=0.01, seed=42)
        assert abs(tf.reduce_mean(result).numpy() - 5.0) < 0.1

    def test_seed_reproducibility(self):
        """Test that same global + op seed produces same results."""
        tf.random.set_seed(0)
        r1 = random.normal((5,), seed=42)
        tf.random.set_seed(0)
        r2 = random.normal((5,), seed=42)
        np.testing.assert_array_equal(r1.numpy(), r2.numpy())

    def test_seed_generator(self):
        """Test with SeedGenerator."""
        gen = random.SeedGenerator(42)
        result = random.normal((3, 4), seed=gen)
        assert result.shape == (3, 4)


class TestUniform:
    """Test uniform distribution sampling."""

    def test_shape(self):
        """Test output shape."""
        result = random.uniform((3, 4))
        assert result.shape == (3, 4)

    def test_range(self):
        """Test values are within range."""
        result = random.uniform((1000,), minval=2.0, maxval=5.0)
        assert tf.reduce_all(result >= 2.0).numpy()
        assert tf.reduce_all(result < 5.0).numpy()

    def test_dtype(self):
        """Test output dtype."""
        result = random.uniform((3,), dtype="float32")
        assert result.dtype == tf.float32

    def test_seed_reproducibility(self):
        """Test that same global + op seed produces same results."""
        tf.random.set_seed(0)
        r1 = random.uniform((5,), seed=42)
        tf.random.set_seed(0)
        r2 = random.uniform((5,), seed=42)
        np.testing.assert_array_equal(r1.numpy(), r2.numpy())


class TestTruncatedNormal:
    """Test truncated normal distribution sampling."""

    def test_shape(self):
        """Test output shape."""
        result = random.truncated_normal((3, 4))
        assert result.shape == (3, 4)

    def test_within_bounds(self):
        """Test values are within 2 stddev."""
        result = random.truncated_normal((10000,), mean=0.0, stddev=1.0, seed=42)
        assert tf.reduce_all(result >= -2.0).numpy()
        assert tf.reduce_all(result <= 2.0).numpy()

    def test_seed_reproducibility(self):
        """Test that same global + op seed produces same results."""
        tf.random.set_seed(0)
        r1 = random.truncated_normal((5,), seed=42)
        tf.random.set_seed(0)
        r2 = random.truncated_normal((5,), seed=42)
        np.testing.assert_array_equal(r1.numpy(), r2.numpy())


class TestCategorical:
    """Test categorical distribution sampling."""

    def test_shape(self):
        """Test output shape."""
        logits = tf.constant([[0.1, 0.9, 0.0]])
        result = random.categorical(logits, num_samples=5)
        assert result.shape == (1, 5)

    def test_dtype(self):
        """Test output dtype."""
        logits = tf.constant([[0.1, 0.9, 0.0]])
        result = random.categorical(logits, num_samples=5, dtype="int32")
        assert result.dtype == tf.int32

    def test_batch(self):
        """Test batch categorical sampling."""
        logits = tf.constant([[0.1, 0.9], [0.9, 0.1]])
        result = random.categorical(logits, num_samples=3)
        assert result.shape == (2, 3)


class TestRandint:
    """Test random integer generation."""

    def test_shape(self):
        """Test output shape."""
        result = random.randint((3, 4), minval=0, maxval=10)
        assert result.shape == (3, 4)

    def test_dtype(self):
        """Test output dtype."""
        result = random.randint((3,), minval=0, maxval=10, dtype="int32")
        assert result.dtype == tf.int32

    def test_range(self):
        """Test values are within range."""
        result = random.randint((1000,), minval=0, maxval=10)
        assert tf.reduce_all(result >= 0).numpy()
        assert tf.reduce_all(result < 10).numpy()

    def test_seed_reproducibility(self):
        """Test that same global + op seed produces same results."""
        tf.random.set_seed(0)
        r1 = random.randint((5,), minval=0, maxval=100, seed=42)
        tf.random.set_seed(0)
        r2 = random.randint((5,), minval=0, maxval=100, seed=42)
        np.testing.assert_array_equal(r1.numpy(), r2.numpy())


class TestShuffle:
    """Test shuffle operation."""

    def test_shape_preserved(self):
        """Test output shape matches input."""
        x = tf.constant([1, 2, 3, 4, 5])
        result = random.shuffle(x)
        assert result.shape == x.shape

    def test_values_preserved(self):
        """Test that all values are preserved after shuffle."""
        x = tf.constant([1, 2, 3, 4, 5])
        result = random.shuffle(x)
        np.testing.assert_array_equal(
            sorted(result.numpy().tolist()),
            sorted(x.numpy().tolist())
        )

    def test_2d(self):
        """Test shuffle of 2D tensor along axis 0."""
        x = tf.constant([[1, 2], [3, 4], [5, 6]])
        result = random.shuffle(x, axis=0)
        assert result.shape == x.shape


class TestGamma:
    """Test gamma distribution sampling."""

    def test_shape(self):
        """Test output shape."""
        result = random.gamma((3, 4), alpha=2.0)
        assert result.shape == (3, 4)

    def test_positive_values(self):
        """Test that gamma samples are positive."""
        result = random.gamma((100,), alpha=2.0)
        assert tf.reduce_all(result > 0).numpy()

    def test_dtype(self):
        """Test output dtype."""
        result = random.gamma((3,), alpha=2.0, dtype="float32")
        assert result.dtype == tf.float32


class TestBinomial:
    """Test binomial distribution sampling."""

    def test_shape(self):
        """Test output shape."""
        result = random.binomial((3, 4), counts=10, probabilities=0.5)
        assert result.shape == (3, 4)

    def test_range(self):
        """Test values are within [0, counts]."""
        result = random.binomial((100,), counts=10, probabilities=0.5)
        assert tf.reduce_all(result >= 0).numpy()
        assert tf.reduce_all(result <= 10).numpy()


class TestBeta:
    """Test beta distribution sampling."""

    def test_shape(self):
        """Test output shape."""
        result = random.beta((3, 4), alpha=2.0, beta=5.0)
        assert result.shape == (3, 4)

    def test_range(self):
        """Test values are within [0, 1]."""
        result = random.beta((1000,), alpha=2.0, beta=5.0)
        assert tf.reduce_all(result >= 0.0).numpy()
        assert tf.reduce_all(result <= 1.0).numpy()

    def test_dtype(self):
        """Test output dtype."""
        result = random.beta((3,), alpha=2.0, beta=5.0, dtype="float32")
        assert result.dtype == tf.float32


class TestDropout:
    """Test dropout operation."""

    def test_shape_preserved(self):
        """Test output shape matches input."""
        x = tf.ones((3, 4))
        result = random.dropout(x, rate=0.5)
        assert result.shape == x.shape

    def test_zero_rate(self):
        """Test dropout with rate=0 keeps all values."""
        x = tf.ones((3, 4))
        result = random.dropout(x, rate=0.0)
        np.testing.assert_array_equal(result.numpy(), x.numpy())

    def test_values_scaled(self):
        """Test that surviving values are scaled appropriately."""
        tf.random.set_seed(42)
        x = tf.ones((1000,))
        result = random.dropout(x, rate=0.5, seed=42)
        # Non-zero values should be scaled by 1/(1-rate) = 2.0
        non_zero = result[result != 0]
        if len(non_zero) > 0:
            np.testing.assert_array_almost_equal(
                non_zero.numpy(),
                np.full(non_zero.shape, 2.0),
                decimal=5
            )


class TestModuleExports:
    """Test module exports and accessibility."""

    def test_all_exports(self):
        """Test __all__ contains expected exports."""
        expected = [
            "SeedGenerator", "beta", "binomial", "categorical", "dropout",
            "gamma", "normal", "randint", "shuffle", "truncated_normal", "uniform",
        ]
        for name in expected:
            assert name in random.__all__

    def test_all_callable(self):
        """Test all exported functions are callable."""
        for name in random.__all__:
            obj = getattr(random, name)
            assert callable(obj)
