"""Type stubs for tensorflow.tpu module - provided by LegacyKerasPatch."""

from typing import Any

from . import experimental as experimental

def __getattr__(name: str) -> Any: ...
