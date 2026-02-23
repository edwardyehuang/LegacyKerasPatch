"""Type stubs for tensorflow.data.experimental module - provided by LegacyKerasPatch."""

from typing import Any

from tensorflow.python.data.ops.debug_mode import toggle_debug_mode as enable_debug_mode

from tensorflow.python.data.ops.options import AutoShardPolicy as AutoShardPolicy
from tensorflow.python.data.ops.options import AutotuneAlgorithm as AutotuneAlgorithm
from tensorflow.python.data.ops.options import AutotuneOptions as AutotuneOptions
from tensorflow.python.data.ops.options import DistributeOptions as DistributeOptions
from tensorflow.python.data.ops.options import ExternalStatePolicy as ExternalStatePolicy
from tensorflow.python.data.ops.options import OptimizationOptions as OptimizationOptions
from tensorflow.python.data.ops.options import ServiceOptions as ServiceOptions
from tensorflow.python.data.ops.options import ThreadingOptions as ThreadingOptions

from tensorflow.python.data.ops.optional_ops import Optional as Optional

from tensorflow.python.data.experimental.ops.counter import CounterV2 as Counter
from tensorflow.python.data.experimental.ops.readers import CsvDatasetV2 as CsvDataset
from tensorflow.python.data.experimental.ops.readers import SqlDatasetV2 as SqlDataset
from tensorflow.python.data.experimental.ops.lookup_ops import DatasetInitializer as DatasetInitializer
from tensorflow.python.data.experimental.ops.random_ops import RandomDatasetV2 as RandomDataset
from tensorflow.python.data.experimental.ops.grouping import Reducer as Reducer
from tensorflow.python.data.experimental.ops.writers import TFRecordWriter as TFRecordWriter

from tensorflow.python.data.experimental.ops.cardinality import assert_cardinality as assert_cardinality
from tensorflow.python.data.experimental.ops.cardinality import cardinality as cardinality
from tensorflow.python.data.experimental.ops.random_access import at as at
from tensorflow.python.data.experimental.ops.grouping import bucket_by_sequence_length as bucket_by_sequence_length
from tensorflow.python.data.experimental.ops.grouping import group_by_reducer as group_by_reducer
from tensorflow.python.data.experimental.ops.grouping import group_by_window as group_by_window
from tensorflow.python.data.experimental.ops.interleave_ops import choose_from_datasets as choose_from_datasets
from tensorflow.python.data.experimental.ops.interleave_ops import sample_from_datasets as sample_from_datasets
from tensorflow.python.data.experimental.ops.batching import dense_to_ragged_batch as dense_to_ragged_batch
from tensorflow.python.data.experimental.ops.batching import dense_to_sparse_batch as dense_to_sparse_batch
from tensorflow.python.data.experimental.ops.batching import map_and_batch as map_and_batch
from tensorflow.python.data.experimental.ops.batching import unbatch as unbatch
from tensorflow.python.data.experimental.ops.enumerate_ops import enumerate_dataset as enumerate_dataset
from tensorflow.python.data.experimental.ops.error_ops import ignore_errors as ignore_errors
from tensorflow.python.data.experimental.ops.from_list import from_list as from_list
from tensorflow.python.data.experimental.ops.get_single_element import get_single_element as get_single_element
from tensorflow.python.data.experimental.ops.io import load as load
from tensorflow.python.data.experimental.ops.io import save as save
from tensorflow.python.data.experimental.ops.parsing_ops import parse_example_dataset as parse_example_dataset
from tensorflow.python.data.experimental.ops.prefetching_ops import copy_to_device as copy_to_device
from tensorflow.python.data.experimental.ops.prefetching_ops import prefetch_to_device as prefetch_to_device
from tensorflow.python.data.experimental.ops.resampling import rejection_resample as rejection_resample
from tensorflow.python.data.experimental.ops.scan_ops import scan as scan
from tensorflow.python.data.experimental.ops.shuffle_ops import shuffle_and_repeat as shuffle_and_repeat
from tensorflow.python.data.experimental.ops.snapshot import snapshot as snapshot
from tensorflow.python.data.experimental.ops.take_while_ops import take_while as take_while
from tensorflow.python.data.experimental.ops.unique import unique as unique
from tensorflow.python.data.experimental.ops.readers import make_batched_features_dataset as make_batched_features_dataset
from tensorflow.python.data.experimental.ops.readers import make_csv_dataset as make_csv_dataset

from tensorflow.python.data.ops.dataset_ops import from_variant as from_variant
from tensorflow.python.data.ops.dataset_ops import get_structure as get_structure
from tensorflow.python.data.ops.dataset_ops import to_variant as to_variant
from tensorflow.python.data.ops.iterator_ops import get_next_as_optional as get_next_as_optional

AUTOTUNE: int
INFINITE_CARDINALITY: int
SHARD_HINT: int
UNKNOWN_CARDINALITY: int

def __getattr__(name: str) -> Any: ...
