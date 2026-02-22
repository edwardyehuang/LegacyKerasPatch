"""Type stubs for tensorflow.tpu.experimental module - provided by LegacyKerasPatch."""

from typing import Any

from tensorflow.python.distribute.cluster_resolver.tpu.tpu_cluster_resolver import initialize_tpu_system as initialize_tpu_system
from tensorflow.python.distribute.cluster_resolver.tpu.tpu_cluster_resolver import shutdown_tpu_system as shutdown_tpu_system
from tensorflow.python.tpu.device_assignment import DeviceAssignment as DeviceAssignment
from tensorflow.python.tpu.device_assignment import DeviceOrderMode as DeviceOrderMode
from tensorflow.python.tpu.tpu_hardware_feature import HardwareFeature as HardwareFeature
from tensorflow.python.tpu.tpu_system_metadata import TPUSystemMetadata as TPUSystemMetadata
from tensorflow.python.tpu.topology import Topology as Topology

def __getattr__(name: str) -> Any: ...
