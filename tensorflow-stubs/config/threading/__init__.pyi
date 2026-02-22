"""Type stubs for tensorflow.config.threading module - provided by LegacyKerasPatch."""

from typing import Any

from tensorflow.python.framework.config import get_inter_op_parallelism_threads as get_inter_op_parallelism_threads
from tensorflow.python.framework.config import get_intra_op_parallelism_threads as get_intra_op_parallelism_threads
from tensorflow.python.framework.config import set_inter_op_parallelism_threads as set_inter_op_parallelism_threads
from tensorflow.python.framework.config import set_intra_op_parallelism_threads as set_intra_op_parallelism_threads

def __getattr__(name: str) -> Any: ...
