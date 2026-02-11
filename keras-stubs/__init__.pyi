"""Type stubs for keras module with ops support from LegacyKerasPatch."""

from typing import Any

from keras import ops as ops

def __getattr__(name: str) -> Any: ...
