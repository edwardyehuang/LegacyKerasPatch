"""Type stubs for tensorflow.distribute.cluster_resolver module - provided by LegacyKerasPatch."""

from typing import Any

from tensorflow.python.distribute.cluster_resolver.cluster_resolver import ClusterResolver as ClusterResolver
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import SimpleClusterResolver as SimpleClusterResolver
from tensorflow.python.distribute.cluster_resolver.tpu.tpu_cluster_resolver import TPUClusterResolver as TPUClusterResolver

class UnionResolver: ...

def __getattr__(name: str) -> Any: ...
