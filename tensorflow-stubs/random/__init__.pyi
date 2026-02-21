"""Type stubs for tensorflow.random module - provided by LegacyKerasPatch."""

from collections.abc import Sequence
from typing import Any

import tensorflow as tf

from . import experimental as experimental

from tensorflow.python.ops.random_ops import random_normal as normal
from tensorflow.python.ops.random_ops import random_uniform as uniform
from tensorflow.python.ops.random_ops import truncated_normal as truncated_normal
from tensorflow.python.ops.random_ops import random_shuffle as shuffle
from tensorflow.python.ops.random_ops import categorical as categorical
from tensorflow.python.ops.random_ops import random_gamma as gamma
from tensorflow.python.ops.random_ops import random_poisson_v2 as poisson

from tensorflow.python.ops.stateless_random_ops import fold_in as fold_in
from tensorflow.python.ops.stateless_random_ops import split as split
from tensorflow.python.ops.stateless_random_ops import stateless_random_binomial as stateless_binomial
from tensorflow.python.ops.stateless_random_ops import stateless_categorical as stateless_categorical
from tensorflow.python.ops.stateless_random_ops import stateless_random_gamma as stateless_gamma
from tensorflow.python.ops.stateless_random_ops import stateless_random_normal as stateless_normal
from tensorflow.python.ops.stateless_random_ops import stateless_parameterized_truncated_normal as stateless_parameterized_truncated_normal
from tensorflow.python.ops.stateless_random_ops import stateless_random_poisson as stateless_poisson
from tensorflow.python.ops.stateless_random_ops import stateless_truncated_normal as stateless_truncated_normal
from tensorflow.python.ops.stateless_random_ops import stateless_random_uniform as stateless_uniform

from tensorflow.python.ops.candidate_sampling_ops import all_candidate_sampler as all_candidate_sampler
from tensorflow.python.ops.candidate_sampling_ops import fixed_unigram_candidate_sampler as fixed_unigram_candidate_sampler
from tensorflow.python.ops.candidate_sampling_ops import learned_unigram_candidate_sampler as learned_unigram_candidate_sampler
from tensorflow.python.ops.candidate_sampling_ops import log_uniform_candidate_sampler as log_uniform_candidate_sampler
from tensorflow.python.ops.candidate_sampling_ops import uniform_candidate_sampler as uniform_candidate_sampler

from tensorflow.python.ops.stateful_random_ops import create_rng_state as create_rng_state
from tensorflow.python.ops.stateful_random_ops import get_global_generator as get_global_generator
from tensorflow.python.ops.stateful_random_ops import set_global_generator as set_global_generator

from tensorflow.python.framework.random_seed import set_seed as set_seed

class Algorithm:
    AUTO_SELECT: Any
    PHILOX: Any
    THREEFRY: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class Generator:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def make_seeds(self, count: int = ...) -> tf.Tensor: ...
    def normal(self, shape: tf.Tensor | Sequence[int], mean: float = ..., stddev: float = ..., dtype: Any = ..., name: str | None = ...) -> tf.Tensor: ...
    def split(self, count: int = ...) -> list[Generator]: ...
    def truncated_normal(self, shape: Any, mean: float = ..., stddev: float = ..., dtype: Any = ..., name: str | None = ...) -> tf.Tensor: ...
    def uniform(self, shape: Any, minval: float = ..., maxval: float | None = ..., dtype: Any = ..., name: str | None = ...) -> tf.Tensor: ...
    @staticmethod
    def from_seed(seed: int, alg: Any = ...) -> Generator: ...
    @staticmethod
    def from_non_deterministic_state(alg: Any = ...) -> Generator: ...
    @staticmethod
    def from_key_counter(key: Any, counter: Any, alg: Any = ...) -> Generator: ...

def __getattr__(name: str) -> Any: ...
