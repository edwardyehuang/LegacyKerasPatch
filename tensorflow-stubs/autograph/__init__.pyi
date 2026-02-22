"""Type stubs for tensorflow.autograph module - provided by LegacyKerasPatch."""

from typing import Any

from . import experimental as experimental

from tensorflow.python.autograph.impl.api import to_graph as to_graph
from tensorflow.python.autograph.impl.api import to_code as to_code

from tensorflow.python.autograph.utils.ag_logging import set_verbosity as set_verbosity
from tensorflow.python.autograph.utils.ag_logging import trace as trace

def __getattr__(name: str) -> Any: ...
