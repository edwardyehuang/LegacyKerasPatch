"""Type stubs for tensorflow.experimental.dtensor module - provided by LegacyKerasPatch."""

from typing import Any

# Layout and Mesh classes from tensorflow.dtensor.python.layout
from tensorflow.dtensor.python.layout import Layout as Layout
from tensorflow.dtensor.python.layout import Mesh as Mesh

# Constants from tensorflow.dtensor.python.layout
MATCH: str
UNSHARDED: str

# DTensorCheckpoint class
from tensorflow.dtensor.python.d_checkpoint import DTensorCheckpoint as DTensorCheckpoint

# DTensorDataset
from tensorflow.dtensor.python.input_util import DTensorDataset as DTensorDataset

# DVariable
from tensorflow.dtensor.python.d_variable import DVariable as DVariable

# Functions from tensorflow.dtensor.python.api
from tensorflow.dtensor.python.api import call_with_layout as call_with_layout
from tensorflow.dtensor.python.api import check_layout as check_layout
from tensorflow.dtensor.python.api import copy_to_mesh as copy_to_mesh
from tensorflow.dtensor.python.api import default_mesh as default_mesh
from tensorflow.dtensor.python.api import device_name as device_name
from tensorflow.dtensor.python.api import fetch_layout as fetch_layout
from tensorflow.dtensor.python.api import get_default_mesh as get_default_mesh
from tensorflow.dtensor.python.api import is_dtensor as is_dtensor
from tensorflow.dtensor.python.api import pack as pack
from tensorflow.dtensor.python.api import relayout as relayout
from tensorflow.dtensor.python.api import relayout_like as relayout_like
from tensorflow.dtensor.python.api import run_on as run_on
from tensorflow.dtensor.python.api import unpack as unpack

# Functions from tensorflow.dtensor.python.mesh_util
from tensorflow.dtensor.python.mesh_util import barrier as barrier
from tensorflow.dtensor.python.mesh_util import create_distributed_mesh as create_distributed_mesh
from tensorflow.dtensor.python.mesh_util import create_mesh as create_mesh

# Functions from tensorflow.dtensor.python.config
from tensorflow.dtensor.python.config import client_id as client_id
from tensorflow.dtensor.python.config import full_job_name as full_job_name
from tensorflow.dtensor.python.config import heartbeat_enabled as heartbeat_enabled
from tensorflow.dtensor.python.config import job_name as job_name
from tensorflow.dtensor.python.config import jobs as jobs
from tensorflow.dtensor.python.config import local_devices as local_devices
from tensorflow.dtensor.python.config import num_clients as num_clients
from tensorflow.dtensor.python.config import num_global_devices as num_global_devices
from tensorflow.dtensor.python.config import num_local_devices as num_local_devices
from tensorflow.dtensor.python.config import preferred_device_type as preferred_device_type

# Functions from tensorflow.dtensor.python.accelerator_util
from tensorflow.dtensor.python.accelerator_util import initialize_accelerator_system as initialize_accelerator_system
from tensorflow.dtensor.python.accelerator_util import initialize_accelerator_system as initialize_multi_client
from tensorflow.dtensor.python.accelerator_util import shutdown_accelerator_system as shutdown_accelerator_system

# Functions from tensorflow.dtensor.python.tpu_util
from tensorflow.dtensor.python.tpu_util import create_tpu_mesh as create_tpu_mesh
from tensorflow.dtensor.python.tpu_util import initialize_tpu_system as initialize_tpu_system
from tensorflow.dtensor.python.tpu_util import shutdown_tpu_system as shutdown_tpu_system

# Functions from tensorflow.dtensor.python.save_restore
from tensorflow.dtensor.python.save_restore import enable_save_as_bf16 as enable_save_as_bf16
from tensorflow.dtensor.python.save_restore import name_based_restore as name_based_restore
from tensorflow.dtensor.python.save_restore import name_based_save as name_based_save
from tensorflow.dtensor.python.save_restore import sharded_save as sharded_save

def __getattr__(name: str) -> Any: ...
