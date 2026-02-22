"""Type stubs for tensorflow.experimental module - provided by LegacyKerasPatch."""

from typing import Any

from . import dtensor as dtensor
from . import numpy as numpy

def __getattr__(name: str) -> Any: ...
