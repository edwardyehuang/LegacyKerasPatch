"""Type stubs for tensorflow.distribute module - provided by LegacyKerasPatch."""

from typing import Any

from tensorflow.python.distribute.tpu_strategy import TPUStrategyV2 as TPUStrategy
from tensorflow.python.distribute.mirrored_strategy import MirroredStrategy as MirroredStrategy
from tensorflow.python.distribute.collective_all_reduce_strategy import CollectiveAllReduceStrategy as MultiWorkerMirroredStrategy
from tensorflow.python.distribute.one_device_strategy import OneDeviceStrategy as OneDeviceStrategy
from tensorflow.python.distribute.parameter_server_strategy_v2 import ParameterServerStrategyV2 as ParameterServerStrategy
from tensorflow.python.distribute.distribute_lib import Strategy as Strategy
from tensorflow.python.distribute.distribute_lib import StrategyExtended as StrategyExtended
from tensorflow.python.distribute.distribute_lib import ReplicaContext as ReplicaContext
from tensorflow.python.distribute.distribute_lib import InputContext as InputContext
from tensorflow.python.distribute.distribute_lib import InputOptions as InputOptions
from tensorflow.python.distribute.distribute_lib import InputReplicationMode as InputReplicationMode
from tensorflow.python.distribute.distribute_lib import RunOptions as RunOptions
from tensorflow.python.distribute.distribute_lib import experimental_set_strategy as experimental_set_strategy
from tensorflow.python.distribute.distribute_lib import get_replica_context as get_replica_context
from tensorflow.python.distribute.distribute_lib import get_strategy as get_strategy
from tensorflow.python.distribute.distribute_lib import has_strategy as has_strategy
from tensorflow.python.distribute.distribute_lib import in_cross_replica_context as in_cross_replica_context
from tensorflow.python.distribute.cross_device_ops import CrossDeviceOps as CrossDeviceOps
from tensorflow.python.distribute.cross_device_ops import HierarchicalCopyAllReduce as HierarchicalCopyAllReduce
from tensorflow.python.distribute.cross_device_ops import NcclAllReduce as NcclAllReduce
from tensorflow.python.distribute.cross_device_ops import ReductionToOneDevice as ReductionToOneDevice
from tensorflow.python.distribute.reduce_util import ReduceOp as ReduceOp
from tensorflow.python.training.server_lib import Server as Server

from . import cluster_resolver as cluster_resolver
from . import coordinator as coordinator
from . import experimental as experimental

def __getattr__(name: str) -> Any: ...
