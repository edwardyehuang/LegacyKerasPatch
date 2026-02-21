"""Type stubs for tensorflow.autograph.experimental module - provided by LegacyKerasPatch."""

from typing import Any

def do_not_convert(func: Any = ...) -> Any: ...

class Feature:
    ALL: Any
    AUTO_CONTROL_DEPS: Any
    ASSERT_STATEMENTS: Any
    BUILTIN_FUNCTIONS: Any
    EQUALITY_OPERATORS: Any
    LISTS: Any
    NAME_SCOPES: Any

def __getattr__(name: str) -> Any: ...
