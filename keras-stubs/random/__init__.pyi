"""Type stubs for keras.random module - provided by LegacyKerasPatch."""

from legacy_keras_patch.random import *
from legacy_keras_patch.random import SeedGenerator as SeedGenerator

def __getattr__(name: str) -> Any: ...

from typing import Any
