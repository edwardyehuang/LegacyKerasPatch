"""
Keras 3 random compatibility layer for Keras 2.

This module provides keras.random compatible operations by wrapping TensorFlow functions,
enabling Keras 2 users to migrate their code to use keras.random in preparation for Keras 3.
"""

import tensorflow as tf

__all__ = [
    "SeedGenerator",
    "beta",
    "binomial",
    "categorical",
    "dropout",
    "gamma",
    "normal",
    "randint",
    "shuffle",
    "truncated_normal",
    "uniform",
]


class SeedGenerator:
    """A class to generate seeds for random operations.

    This is a compatibility shim for keras.random.SeedGenerator in Keras 3.
    It wraps a seed value and provides a consistent interface for seeding
    random operations.

    Args:
        seed: Initial seed value. If None, a random seed is used.
    """

    def __init__(self, seed=None):
        if seed is None:
            self._seed = int(tf.random.uniform([], minval=0, maxval=2**31 - 1, dtype=tf.int32).numpy())
        elif isinstance(seed, SeedGenerator):
            self._seed = seed._seed
        else:
            self._seed = int(seed)
        self._counter = 0

    def next(self, count=1):
        """Get the next seed(s).

        Args:
            count: Number of seeds to generate.

        Returns:
            A list of integer seeds.
        """
        seeds = []
        for _ in range(count):
            self._counter += 1
            seeds.append(self._seed + self._counter)
        if count == 1:
            return seeds[0]
        return seeds


def _get_seed(seed):
    """Extract an integer seed from a seed argument.

    Args:
        seed: An integer, a SeedGenerator, or None.

    Returns:
        An integer seed or None.
    """
    if seed is None:
        return None
    if isinstance(seed, SeedGenerator):
        return seed.next()
    return int(seed)


def normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    """Draw random samples from a normal (Gaussian) distribution.

    Args:
        shape: The shape of the random tensor.
        mean: Mean of the distribution. Defaults to 0.0.
        stddev: Standard deviation of the distribution. Defaults to 1.0.
        dtype: The data type of the output tensor. Defaults to "float32".
        seed: Optional seed for reproducibility.

    Returns:
        A tensor of random values drawn from a normal distribution.
    """
    if dtype is None:
        dtype = "float32"
    seed = _get_seed(seed)
    return tf.random.normal(shape, mean=mean, stddev=stddev, dtype=dtype, seed=seed)


def uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    """Draw random samples from a uniform distribution.

    Args:
        shape: The shape of the random tensor.
        minval: Lower bound of the distribution. Defaults to 0.0.
        maxval: Upper bound of the distribution. Defaults to 1.0.
        dtype: The data type of the output tensor. Defaults to "float32".
        seed: Optional seed for reproducibility.

    Returns:
        A tensor of random values drawn from a uniform distribution.
    """
    if dtype is None:
        dtype = "float32"
    seed = _get_seed(seed)
    return tf.random.uniform(shape, minval=minval, maxval=maxval, dtype=dtype, seed=seed)


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    """Draw random samples from a truncated normal distribution.

    The values are drawn from a normal distribution with the given mean and
    standard deviation, discarding and re-drawing any samples more than two
    standard deviations from the mean.

    Args:
        shape: The shape of the random tensor.
        mean: Mean of the distribution. Defaults to 0.0.
        stddev: Standard deviation of the distribution. Defaults to 1.0.
        dtype: The data type of the output tensor. Defaults to "float32".
        seed: Optional seed for reproducibility.

    Returns:
        A tensor of random values drawn from a truncated normal distribution.
    """
    if dtype is None:
        dtype = "float32"
    seed = _get_seed(seed)
    return tf.random.truncated_normal(shape, mean=mean, stddev=stddev, dtype=dtype, seed=seed)


def categorical(logits, num_samples, dtype=None, seed=None):
    """Draw random samples from a categorical distribution.

    Args:
        logits: 2-D tensor of log-odds (unnormalized log probabilities).
        num_samples: Number of samples to draw for each row of logits.
        dtype: The data type of the output tensor. Defaults to "int32".
        seed: Optional seed for reproducibility.

    Returns:
        A 2-D tensor of sampled category indices.
    """
    if dtype is None:
        dtype = "int32"
    seed = _get_seed(seed)
    result = tf.random.categorical(logits, num_samples, seed=seed)
    return tf.cast(result, dtype)


def randint(shape, minval, maxval, dtype=None, seed=None):
    """Draw random integers from a uniform distribution.

    Args:
        shape: The shape of the random tensor.
        minval: Lower bound (inclusive).
        maxval: Upper bound (exclusive).
        dtype: The data type of the output tensor. Defaults to "int32".
        seed: Optional seed for reproducibility.

    Returns:
        A tensor of random integers.
    """
    if dtype is None:
        dtype = "int32"
    seed = _get_seed(seed)
    return tf.random.uniform(shape, minval=minval, maxval=maxval, dtype=dtype, seed=seed)


def shuffle(x, axis=0, seed=None):
    """Shuffle a tensor along a given axis.

    Args:
        x: The tensor to shuffle.
        axis: The axis along which to shuffle. Defaults to 0.
        seed: Optional seed for reproducibility.

    Returns:
        The shuffled tensor.
    """
    seed = _get_seed(seed)
    if seed is not None:
        tf.random.set_seed(seed)
    if axis == 0:
        return tf.random.shuffle(x, seed=seed)
    # For other axes, transpose, shuffle, transpose back
    perm = list(range(len(x.shape)))
    perm[0], perm[axis] = perm[axis], perm[0]
    x_t = tf.transpose(x, perm)
    x_t = tf.random.shuffle(x_t, seed=seed)
    return tf.transpose(x_t, perm)


def gamma(shape, alpha, dtype=None, seed=None):
    """Draw random samples from a gamma distribution.

    Args:
        shape: The shape of the random tensor.
        alpha: Shape parameter of the gamma distribution (must be > 0).
        dtype: The data type of the output tensor. Defaults to "float32".
        seed: Optional seed for reproducibility.

    Returns:
        A tensor of random values drawn from a gamma distribution.
    """
    if dtype is None:
        dtype = "float32"
    seed = _get_seed(seed)
    return tf.random.gamma(shape, alpha=alpha, dtype=dtype, seed=seed)


def binomial(shape, counts, probabilities, dtype=None, seed=None):
    """Draw random samples from a binomial distribution.

    Args:
        shape: The shape of the random tensor.
        counts: Number of trials (n parameter).
        probabilities: Probability of success (p parameter).
        dtype: The data type of the output tensor. Defaults to "float32".
        seed: Optional seed for reproducibility.

    Returns:
        A tensor of random values drawn from a binomial distribution.
    """
    if dtype is None:
        dtype = "float32"
    seed_val = _get_seed(seed)
    # Simulate binomial using sum of Bernoulli trials
    counts = tf.cast(counts, tf.int32)
    probabilities = tf.cast(probabilities, tf.float32)
    max_count = tf.reduce_max(counts)
    # Generate uniform samples for each trial
    uniform_samples = tf.random.uniform(
        tf.concat([shape, [max_count]], axis=0),
        dtype=tf.float32,
        seed=seed_val,
    )
    # Count successes
    successes = tf.cast(uniform_samples < probabilities, tf.float32)
    # Mask out trials beyond counts
    trial_indices = tf.range(max_count)
    mask = tf.cast(trial_indices < counts, tf.float32)
    successes = successes * mask
    result = tf.reduce_sum(successes, axis=-1)
    return tf.cast(result, dtype)


def beta(shape, alpha, beta, dtype=None, seed=None):
    """Draw random samples from a beta distribution.

    Uses the relationship between beta and gamma distributions:
    If X ~ Gamma(alpha) and Y ~ Gamma(beta), then X/(X+Y) ~ Beta(alpha, beta).

    Args:
        shape: The shape of the random tensor.
        alpha: Alpha parameter of the beta distribution (must be > 0).
        beta: Beta parameter of the beta distribution (must be > 0).
        dtype: The data type of the output tensor. Defaults to "float32".
        seed: Optional seed for reproducibility.

    Returns:
        A tensor of random values drawn from a beta distribution.
    """
    if dtype is None:
        dtype = "float32"
    seed_val = _get_seed(seed)
    x = tf.random.gamma(shape, alpha=alpha, dtype=dtype, seed=seed_val)
    seed_val2 = seed_val + 1 if seed_val is not None else None
    y = tf.random.gamma(shape, alpha=beta, dtype=dtype, seed=seed_val2)
    return x / (x + y)


def dropout(inputs, rate, noise_shape=None, seed=None):
    """Apply dropout to the input tensor.

    Args:
        inputs: The input tensor.
        rate: Fraction of the input values to drop (between 0 and 1).
        noise_shape: Shape of the binary dropout mask that will be
            multiplied with the input.
        seed: Optional seed for reproducibility.

    Returns:
        The input tensor with dropout applied.
    """
    seed = _get_seed(seed)
    return tf.nn.dropout(inputs, rate=rate, noise_shape=noise_shape, seed=seed)
