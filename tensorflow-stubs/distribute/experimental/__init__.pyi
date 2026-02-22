"""Type stubs for tensorflow.distribute.experimental module - provided by LegacyKerasPatch."""

from typing import Any

from tensorflow.python.distribute.central_storage_strategy import CentralStorageStrategy as CentralStorageStrategy
from tensorflow.python.distribute.collective_util import CollectiveCommunication as CollectiveCommunication
from tensorflow.python.distribute.collective_util import CommunicationImplementation as CommunicationImplementation
from tensorflow.python.distribute.distribute_lib import ValueContext as ValueContext
from tensorflow.python.distribute.tpu_strategy import TPUStrategyV2 as TPUStrategy

class CollectiveHints: ...
class CommunicationOptions: ...

def __getattr__(name: str) -> Any: ...
