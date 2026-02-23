"""Type stubs for tensorflow.data module - provided by LegacyKerasPatch."""

from typing import Any

from . import experimental as experimental

from tensorflow.python.data.ops.dataset_ops import DatasetV2 as Dataset
from tensorflow.python.data.ops.dataset_ops import DatasetSpec as DatasetSpec
from tensorflow.python.data.ops.dataset_ops import NumpyIterator as NumpyIterator

from tensorflow.python.data.ops.iterator_ops import OwnedIterator as Iterator
from tensorflow.python.data.ops.iterator_ops import IteratorSpec as IteratorSpec

from tensorflow.python.data.ops.readers import TFRecordDatasetV2 as TFRecordDataset
from tensorflow.python.data.ops.readers import TextLineDatasetV2 as TextLineDataset
from tensorflow.python.data.ops.readers import FixedLengthRecordDatasetV2 as FixedLengthRecordDataset

from tensorflow.python.data.ops.options import Options as Options
from tensorflow.python.data.ops.options import ThreadingOptions as ThreadingOptions

AUTOTUNE: int
INFINITE_CARDINALITY: int
UNKNOWN_CARDINALITY: int

def __getattr__(name: str) -> Any: ...
