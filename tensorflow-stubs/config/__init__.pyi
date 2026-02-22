"""Type stubs for tensorflow.config module - provided by LegacyKerasPatch."""

from typing import Any

from . import experimental as experimental
from . import optimizer as optimizer
from . import threading as threading

from tensorflow.python.eager.context import LogicalDevice as LogicalDevice
from tensorflow.python.eager.context import LogicalDeviceConfiguration as LogicalDeviceConfiguration
from tensorflow.python.eager.context import PhysicalDevice as PhysicalDevice

from tensorflow.python.eager.polymorphic_function.eager_function_run import run_functions_eagerly as run_functions_eagerly
from tensorflow.python.eager.polymorphic_function.eager_function_run import functions_run_eagerly as functions_run_eagerly
from tensorflow.python.eager.polymorphic_function.eager_function_run import experimental_run_functions_eagerly as experimental_run_functions_eagerly
from tensorflow.python.eager.polymorphic_function.eager_function_run import experimental_functions_run_eagerly as experimental_functions_run_eagerly

from tensorflow.python.eager.remote import connect_to_cluster as experimental_connect_to_cluster
from tensorflow.python.eager.remote import connect_to_remote_host as experimental_connect_to_host

from tensorflow.python.framework.config import get_soft_device_placement as get_soft_device_placement
from tensorflow.python.framework.config import set_soft_device_placement as set_soft_device_placement
from tensorflow.python.framework.config import list_physical_devices as list_physical_devices
from tensorflow.python.framework.config import list_logical_devices as list_logical_devices
from tensorflow.python.framework.config import get_visible_devices as get_visible_devices
from tensorflow.python.framework.config import set_visible_devices as set_visible_devices
from tensorflow.python.framework.config import get_logical_device_configuration as get_logical_device_configuration
from tensorflow.python.framework.config import set_logical_device_configuration as set_logical_device_configuration

def __getattr__(name: str) -> Any: ...
