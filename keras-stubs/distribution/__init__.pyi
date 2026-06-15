from typing import Any, Optional, Tuple, List, Sequence, Union
import numpy as np

from legacy_keras_patch.distribution import LayoutMap as LayoutMap

class Distribution:
    device_mesh: Optional["DeviceMesh"]
    batch_dim_name: Optional[str]
    auto_shard_dataset: bool
    num_model_replicas: int
    def get_data_layout(self, data_shape: Tuple[int, ...]) -> "TensorLayout": ...
    def get_variable_layout(self, variable: Any) -> "TensorLayout": ...
    def get_tensor_layout(self, path: str) -> Optional["TensorLayout"]: ...
    def scope(self) -> Any: ...

class DataParallel(Distribution):
    def __init__(
        self,
        device_mesh: Optional["DeviceMesh"] = ...,
        devices: Optional[List[str]] = ...,
        auto_shard_dataset: bool = ...,
    ) -> None: ...

class DeviceMesh:
    shape: Tuple[int, ...]
    axis_names: List[str]
    devices: np.ndarray
    backend_mesh: Any
    def __init__(
        self,
        shape: Sequence[int],
        axis_names: List[str],
        devices: Optional[List[str]] = ...,
    ) -> None: ...

class ModelParallel(Distribution):
    layout_map: Optional[LayoutMap]
    def __init__(
        self,
        device_mesh: Optional["DeviceMesh"] = ...,
        layout_map: Optional[LayoutMap] = ...,
        batch_dim_name: Optional[str] = ...,
        auto_shard_dataset: bool = ...,
    ) -> None: ...

class TensorLayout:
    axes: Tuple[Optional[str], ...]
    device_mesh: Optional[DeviceMesh]
    backend_layout: Any
    is_fully_replicated: bool
    def __init__(
        self,
        axes: Sequence[Optional[str]],
        device_mesh: Optional[DeviceMesh] = ...,
    ) -> None: ...

def distribute_tensor(tensor: Any, layout: Optional["TensorLayout"]) -> Any: ...
def distribute_variable(initial_value: Any, layout: Optional["TensorLayout"], **kwargs: Any) -> Any: ...
def distribute_data_input(data: Any, layout: Optional["TensorLayout"]) -> Any: ...
def distribution() -> Optional[Distribution]: ...
def get_device_count(*args: Any, **kwargs: Any) -> Any: ...
def initialize(*args: Any, **kwargs: Any) -> Any: ...
def list_devices(device_type: Optional[str] = ...) -> List[str]: ...
def set_distribution(value: Optional[Distribution]) -> None: ...

def __getattr__(name: str) -> Any: ...
