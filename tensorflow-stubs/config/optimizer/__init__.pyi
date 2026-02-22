"""Type stubs for tensorflow.config.optimizer module - provided by LegacyKerasPatch."""

from typing import Any

from tensorflow.python.framework.config import get_optimizer_experimental_options as get_experimental_options
from tensorflow.python.framework.config import get_optimizer_jit as get_jit
from tensorflow.python.framework.config import set_optimizer_experimental_options as set_experimental_options
from tensorflow.python.framework.config import set_optimizer_jit as set_jit

def __getattr__(name: str) -> Any: ...
