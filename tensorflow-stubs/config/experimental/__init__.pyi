"""Type stubs for tensorflow.config.experimental module - provided by LegacyKerasPatch."""

from typing import Any

from tensorflow.python.training.server_lib import ClusterDeviceFilters as ClusterDeviceFilters
from tensorflow.python.eager.context import LogicalDeviceConfiguration as VirtualDeviceConfiguration

from tensorflow.python.framework.config import disable_mlir_bridge as disable_mlir_bridge
from tensorflow.python.framework.config import enable_mlir_bridge as enable_mlir_bridge
from tensorflow.python.framework.config import enable_op_determinism as enable_op_determinism
from tensorflow.python.framework.config import enable_tensor_float_32_execution as enable_tensor_float_32_execution
from tensorflow.python.framework.config import get_device_details as get_device_details
from tensorflow.python.framework.config import get_device_policy as get_device_policy
from tensorflow.python.framework.config import get_memory_growth as get_memory_growth
from tensorflow.python.framework.config import get_memory_info as get_memory_info
from tensorflow.python.framework.config import get_memory_usage as get_memory_usage
from tensorflow.python.framework.config import get_synchronous_execution as get_synchronous_execution
from tensorflow.python.framework.config import get_logical_device_configuration as get_virtual_device_configuration
from tensorflow.python.framework.config import get_visible_devices as get_visible_devices
from tensorflow.python.framework.config import list_logical_devices as list_logical_devices
from tensorflow.python.framework.config import list_physical_devices as list_physical_devices
from tensorflow.python.framework.config import reset_memory_stats as reset_memory_stats
from tensorflow.python.framework.config import set_device_policy as set_device_policy
from tensorflow.python.framework.config import set_memory_growth as set_memory_growth
from tensorflow.python.framework.config import set_synchronous_execution as set_synchronous_execution
from tensorflow.python.framework.config import set_logical_device_configuration as set_virtual_device_configuration
from tensorflow.python.framework.config import set_visible_devices as set_visible_devices
from tensorflow.python.framework.config import tensor_float_32_execution_enabled as tensor_float_32_execution_enabled

def __getattr__(name: str) -> Any: ...
