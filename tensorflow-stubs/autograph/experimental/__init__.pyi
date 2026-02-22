"""Type stubs for tensorflow.autograph.experimental module - provided by LegacyKerasPatch."""

from typing import Any

from tensorflow.python.autograph.impl.api import do_not_convert as do_not_convert

from tensorflow.python.autograph.core.converter import Feature as Feature

def __getattr__(name: str) -> Any: ...
