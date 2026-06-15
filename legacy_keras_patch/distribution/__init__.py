"""
Keras 3 distribution compatibility layer for Keras 2.

This module provides keras.distribution compatible classes and functions,
implementing 2D and higher-dimensional mesh support for Keras 2.15 by
bridging to TensorFlow's ``tf.experimental.dtensor`` API.

It supports:
- Multi-dimensional device meshes (2D, 3D, etc.)
- Model parallelism with layout-based variable sharding
- Data parallelism with automatic batch sharding
- Integration with model.fit() and model.evaluate() via distribution scope
- Variable creation with proper dtensor layouts
- Input data distribution across mesh dimensions
"""

import collections
import collections.abc
import contextlib
import math
import re
import threading

import numpy as np
import tensorflow as tf

try:
    from tensorflow.experimental import dtensor as _dtensor
except ImportError:
    _dtensor = None

__all__ = [
    "DataParallel",
    "DeviceMesh",
    "Distribution",
    "LayoutMap",
    "ModelParallel",
    "TensorLayout",
    "distribute_data_input",
    "distribute_tensor",
    "distribute_variable",
    "distribution",
    "list_devices",
    "set_distribution",
]


def list_devices(device_type=None):
    """Return all the available devices based on the device type.

    Args:
        device_type: Optional string of device type, e.g. "CPU" or "GPU".
            Defaults to ``None``, in which case all local devices are returned.

    Returns:
        List of device name strings.
    """
    devices = tf.config.list_logical_devices(device_type)
    return [d.name for d in devices]


# ---------------------------------------------------------------------------
# DeviceMesh
# ---------------------------------------------------------------------------


class DeviceMesh:
    """A cluster of computation devices for distributed computation.

    This API is aligned with ``tf.experimental.dtensor.Mesh`` and supports
    2D and higher-dimensional meshes for combined data + model parallelism.

    Args:
        shape: tuple or list of integers. The shape of the ``DeviceMesh``.
            For example, ``(4, 2)`` creates a 2D mesh with 4 devices along
            the first axis and 2 along the second.
        axis_names: List of strings. The logical name of each axis.
            For example, ``["batch", "model"]``.
        devices: Optional list of devices. Defaults to all available devices.

    Example::

        # 1D mesh for data parallelism
        mesh = DeviceMesh(shape=(8,), axis_names=["batch"])

        # 2D mesh for combined data + model parallelism
        mesh = DeviceMesh(
            shape=(4, 2),
            axis_names=["batch", "model"],
            devices=list_devices("GPU"),
        )
    """

    def __init__(self, shape, axis_names, devices=None):
        if not shape or not axis_names:
            raise ValueError(
                "Shape and axis_names cannot be empty. Received: "
                f"shape={shape}, axis_names={axis_names}"
            )
        if len(shape) != len(axis_names):
            raise ValueError(
                "Shape and axis_names should have same size. "
                f"Received: shape={shape}, axis_names={axis_names}"
            )
        if devices is None:
            devices = list_devices()
        devices = np.array(devices)
        if math.prod(shape) != math.prod(devices.shape):
            raise ValueError(
                "Shape does not match the number of devices. "
                f"Received: shape={shape}; devices.shape="
                f"{devices.shape}"
            )
        self._shape = tuple(shape)
        self._axis_names = list(axis_names)
        self._devices = np.reshape(devices, shape)
        self._backend_mesh = None

    @property
    def shape(self):
        return self._shape

    @property
    def axis_names(self):
        return self._axis_names

    @property
    def devices(self):
        return self._devices

    @property
    def backend_mesh(self):
        """Lazily create and return the dtensor Mesh object.

        Returns:
            A ``tf.experimental.dtensor.Mesh`` instance, or ``None`` if
            dtensor is unavailable.
        """
        if self._backend_mesh is None:
            self._backend_mesh = _to_backend_mesh(self)
        return self._backend_mesh

    def __repr__(self):
        return (
            f"<{self.__class__.__name__} "
            f"shape={self.shape}, axis_names={self.axis_names}>"
        )

    def __str__(self):
        return self.__repr__()


# ---------------------------------------------------------------------------
# TensorLayout
# ---------------------------------------------------------------------------


class TensorLayout:
    """A layout to apply to a tensor.

    This API is aligned with ``tf.experimental.dtensor.Layout``.

    Each element in ``axes`` corresponds to a dimension of the tensor.
    Use a mesh axis name to shard that dimension across the named axis,
    or ``None`` to leave the dimension replicated.

    Args:
        axes: tuple of strings or None values that map to the
            ``axis_names`` in a ``DeviceMesh``.
        device_mesh: Optional ``DeviceMesh`` instance.

    Example::

        # Shard dimension 1 across the "model" axis, replicate dimension 0
        layout = TensorLayout(axes=(None, "model"), device_mesh=mesh)

        # Fully replicated
        layout = TensorLayout(axes=(None, None), device_mesh=mesh)
    """

    def __init__(self, axes, device_mesh=None):
        self._axes = tuple(axes)
        self._device_mesh = device_mesh
        self._backend_layout = None
        self._validate_axes()

    @property
    def axes(self):
        return self._axes

    @property
    def device_mesh(self):
        return self._device_mesh

    @device_mesh.setter
    def device_mesh(self, device_mesh):
        if self._device_mesh is not None:
            raise ValueError(
                "Cannot override device mesh value. Existing "
                f"value is {self._device_mesh}"
            )
        self._device_mesh = device_mesh
        self._backend_layout = None
        self._validate_axes()

    @property
    def backend_layout(self):
        """Lazily create and return the dtensor Layout object.

        Returns:
            A ``tf.experimental.dtensor.Layout`` instance, or ``None`` if
            dtensor is unavailable or device_mesh is not set.
        """
        if self._backend_layout is None and self._device_mesh is not None:
            self._backend_layout = _to_backend_layout(self)
        return self._backend_layout

    @property
    def is_fully_replicated(self):
        """Return True if all axes are None (fully replicated)."""
        return all(a is None for a in self._axes)

    def _validate_axes(self):
        if self._device_mesh:
            valid_axis_names = set(self._device_mesh.axis_names)
            axis_names = set(self._axes) - {None}
            if axis_names - valid_axis_names:
                raise ValueError(
                    "Invalid axis names for Layout. Valid axis "
                    f"names: {valid_axis_names}, Got {axis_names}"
                )

    def __repr__(self):
        return (
            f"<{self.__class__.__name__} "
            f"axes={self.axes}, device_mesh={self.device_mesh}>"
        )

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        if not isinstance(other, TensorLayout):
            return NotImplemented
        return self._axes == other._axes and self._device_mesh is other._device_mesh

    def __hash__(self):
        return hash((self._axes, id(self._device_mesh)))


# ---------------------------------------------------------------------------
# LayoutMap
# ---------------------------------------------------------------------------


class LayoutMap(collections.abc.MutableMapping):
    """A dict-like object that maps string to ``TensorLayout`` instances.

    The string key is treated as a regex when retrieving values.
    This allows pattern-based layout specifications for model
    variables.

    As a shortcut, tuple or list of axis names are also accepted as
    values and will be converted to ``TensorLayout``.

    Example::

        layout_map = LayoutMap(device_mesh)
        layout_map['dense.*kernel'] = (None, 'model')
        layout_map['dense.*bias'] = ('model',)

        layout = layout_map['dense_1.kernel']  # matches via regex

    Args:
        device_mesh: ``DeviceMesh`` instance (also accepted via the
            keyword *mesh* for compatibility with the Keras 2 dtensor
            API).
    """

    def __init__(self, device_mesh=None, *, mesh=None):
        self._layout_map = collections.OrderedDict()
        self._device_mesh = device_mesh if device_mesh is not None else mesh

    def __getitem__(self, key):
        """Retrieve the layout matching *key*.

        If there is no exact match, existing keys are treated as
        regexes.  Returns ``None`` when no match is found.
        """
        if key in self._layout_map:
            return self._layout_map[key]

        matching_keys = []
        for k in self._layout_map:
            if re.search(k, key):
                matching_keys.append(k)
        if len(matching_keys) > 1:
            raise ValueError(
                f"Path '{key}' matches multiple layout "
                f"specification keys: {matching_keys}. Please make "
                "sure each tensor/variable path only matches at most "
                "one layout specification key in the LayoutMap."
            )
        elif len(matching_keys) == 1:
            return self._layout_map[matching_keys[0]]
        return None

    def __setitem__(self, key, layout):
        if key in self._layout_map:
            raise ValueError(
                f"{key} already exist in the LayoutMap with "
                f"value {self._layout_map[key]}. Please make sure to "
                "not use duplicated keys."
            )
        if isinstance(layout, (tuple, list)):
            layout = TensorLayout(axes=layout, device_mesh=None)

        if not isinstance(layout, TensorLayout):
            raise ValueError(
                f"{layout} should be a TensorLayout type, "
                f"got {type(layout)}"
            )
        self._maybe_populate_device_mesh(layout)
        self._layout_map[key] = layout

    def __delitem__(self, key):
        return self._layout_map.pop(key)

    def __len__(self):
        return len(self._layout_map)

    def __iter__(self):
        return iter(self._layout_map)

    @property
    def device_mesh(self):
        return self._device_mesh

    def get_default_mesh(self):
        """Return the default mesh (Keras 2 dtensor compat)."""
        return self._device_mesh

    def _maybe_populate_device_mesh(self, layout):
        if layout.device_mesh is None and self.device_mesh is not None:
            layout.device_mesh = self.device_mesh


# ---------------------------------------------------------------------------
# Distribution base class
# ---------------------------------------------------------------------------


class Distribution:
    """Base class for distribution strategies.

    Provides common interface for ``DataParallel`` and ``ModelParallel``.

    Args:
        device_mesh: A ``DeviceMesh`` instance.
        batch_dim_name: Name of the mesh axis used for data (batch)
            parallelism.
        auto_shard_dataset: Whether to automatically shard datasets.
    """

    def __init__(self, device_mesh=None, batch_dim_name=None,
                 auto_shard_dataset=True):
        self._device_mesh = device_mesh
        self._batch_dim_name = batch_dim_name
        self._auto_shard_dataset = auto_shard_dataset

    @property
    def device_mesh(self):
        return self._device_mesh

    @property
    def batch_dim_name(self):
        return self._batch_dim_name

    @property
    def auto_shard_dataset(self):
        return self._auto_shard_dataset

    @property
    def num_model_replicas(self):
        """Number of model replicas (size of the batch mesh dimension)."""
        if self._device_mesh is None:
            return 1
        if self._batch_dim_name is None:
            return math.prod(self._device_mesh.shape)
        idx = self._device_mesh.axis_names.index(self._batch_dim_name)
        return self._device_mesh.shape[idx]

    def get_data_layout(self, data_shape):
        """Return the ``TensorLayout`` for input data.

        Subclasses should override this to define how input data is
        distributed across devices.

        Args:
            data_shape: Shape tuple of the input data tensor.

        Returns:
            A ``TensorLayout`` instance.
        """
        raise NotImplementedError

    def get_variable_layout(self, variable):
        """Return the ``TensorLayout`` for a model variable.

        Subclasses should override this to define how model weights
        are distributed across devices.

        Args:
            variable: A ``tf.Variable`` or variable-like object with
                ``shape`` and ``name`` attributes.

        Returns:
            A ``TensorLayout`` instance.
        """
        raise NotImplementedError

    def get_tensor_layout(self, path):
        """Return the ``TensorLayout`` for an intermediate tensor.

        Used to specify sharding of layer outputs.

        Args:
            path: String path identifying the tensor (e.g.
                ``"dense_1/output"``).

        Returns:
            A ``TensorLayout`` instance, or ``None``.
        """
        return None

    @contextlib.contextmanager
    def scope(self):
        """Context manager that sets this distribution as the global default.

        Within this scope, variable creation and model compilation will
        use this distribution for layout decisions.

        Example::

            distribution = ModelParallel(layout_map=layout_map)
            with distribution.scope():
                model = keras.Sequential([...])
                model.compile(...)
                model.fit(...)
        """
        original = distribution()
        set_distribution(self)
        try:
            yield
        finally:
            set_distribution(original)


# ---------------------------------------------------------------------------
# DataParallel
# ---------------------------------------------------------------------------


class DataParallel(Distribution):
    """Distribution that replicates model variables and shards data.

    In data parallelism, each device holds a full copy of the model
    and processes a slice of the input batch. This creates a 1D mesh
    if no explicit mesh is provided.

    Args:
        device_mesh: Optional ``DeviceMesh`` instance. If not provided and
            ``devices`` is given, a 1D mesh is created automatically.
        devices: Optional list of device strings. Used to create a default
            1D mesh when ``device_mesh`` is not provided.
        auto_shard_dataset: Whether to auto-shard the dataset. Defaults
            to ``True``.

    Example::

        # Simple data parallel across all GPUs
        dp = DataParallel(device_mesh=DeviceMesh(
            shape=(4,), axis_names=["batch"],
            devices=list_devices("GPU")
        ))
        with dp.scope():
            model.compile(...)
            model.fit(dataset)
    """

    def __init__(self, device_mesh=None, devices=None,
                 auto_shard_dataset=True):
        if device_mesh is None and devices is not None:
            device_mesh = DeviceMesh(
                shape=(len(devices),),
                axis_names=["batch"],
                devices=devices,
            )
        batch_dim_name = "batch"
        if device_mesh is not None and device_mesh.axis_names:
            batch_dim_name = device_mesh.axis_names[0]
        super().__init__(device_mesh, batch_dim_name, auto_shard_dataset)

    def get_data_layout(self, data_shape):
        """Return layout that shards the batch dimension across all devices.

        Args:
            data_shape: Shape tuple of the input data tensor.

        Returns:
            A ``TensorLayout`` with the first dimension sharded across
            the batch axis and all other dimensions replicated.
        """
        data_shard_spec = [None] * len(data_shape)
        if self._batch_dim_name is not None:
            data_shard_spec[0] = self._batch_dim_name
        return TensorLayout(data_shard_spec, self._device_mesh)

    def get_variable_layout(self, variable):
        """Return fully-replicated layout for all variables.

        In data parallelism, all model variables are fully replicated
        across all devices.

        Args:
            variable: A variable with a ``shape`` attribute.

        Returns:
            A fully-replicated ``TensorLayout``.
        """
        if hasattr(variable, '_layout') and variable._layout is not None:
            return variable._layout
        variable_shard_spec = [None] * len(variable.shape)
        return TensorLayout(variable_shard_spec, self._device_mesh)

    def get_tensor_layout(self, path):
        """Data parallelism does not shard intermediate tensors."""
        return None


# ---------------------------------------------------------------------------
# ModelParallel
# ---------------------------------------------------------------------------


class ModelParallel(Distribution):
    """Distribution that shards model variables according to a layout map.

    Model parallelism allows distributing model weights across devices.
    Combined with a multi-dimensional mesh, both data and model parallelism
    can be achieved simultaneously.

    Args:
        device_mesh: Optional ``DeviceMesh`` instance. If not provided,
            it is inferred from ``layout_map.device_mesh``.
        layout_map: A ``LayoutMap`` that specifies how variables should be
            sharded. Required.
        batch_dim_name: The name of the mesh axis used for batch (data)
            parallelism. Defaults to the first axis of the mesh.
        auto_shard_dataset: Whether to auto-shard datasets.

    Example::

        # 2D mesh: 4 batch replicas × 2 model shards
        mesh = DeviceMesh(
            shape=(4, 2),
            axis_names=["batch", "model"],
            devices=list_devices("GPU"),
        )
        layout_map = LayoutMap(mesh)
        layout_map['dense.*kernel'] = (None, 'model')
        layout_map['dense.*bias'] = ('model',)

        mp = ModelParallel(layout_map=layout_map, batch_dim_name="batch")
        with mp.scope():
            model = build_model()
            model.compile(...)
            model.fit(data)
    """

    def __init__(self, device_mesh=None, layout_map=None,
                 batch_dim_name=None, auto_shard_dataset=True):
        if layout_map is not None and device_mesh is None:
            device_mesh = layout_map.device_mesh
        if batch_dim_name is None and device_mesh is not None:
            batch_dim_name = device_mesh.axis_names[0]
        super().__init__(device_mesh, batch_dim_name, auto_shard_dataset)
        self._layout_map = layout_map

    @property
    def layout_map(self):
        return self._layout_map

    def get_data_layout(self, data_shape):
        """Return layout that shards only the batch dimension.

        In model parallelism, input data is sharded along the batch
        dimension (across the ``batch_dim_name`` axis of the mesh).
        Other dimensions remain replicated.

        Args:
            data_shape: Shape tuple of the input data tensor.

        Returns:
            A ``TensorLayout`` with batch dimension sharded.
        """
        data_shard_spec = [None] * len(data_shape)
        if self._batch_dim_name is not None:
            data_shard_spec[0] = self._batch_dim_name
        return TensorLayout(data_shard_spec, self._device_mesh)

    def get_variable_layout(self, variable):
        """Return the layout for a variable based on the layout map.

        Looks up the variable's path/name in the ``LayoutMap`` using
        regex matching. Unmatched variables are fully replicated.

        Args:
            variable: A variable with ``shape`` and ``name`` attributes.

        Returns:
            A ``TensorLayout`` instance.
        """
        if hasattr(variable, '_layout') and variable._layout is not None:
            return variable._layout

        if self._layout_map is not None:
            var_path = _get_variable_path(variable)
            layout = self._layout_map[var_path]
            if layout is not None:
                return layout

        # Default: fully replicate unmatched variables
        variable_shard_spec = [None] * len(variable.shape)
        return TensorLayout(variable_shard_spec, self._device_mesh)

    def get_tensor_layout(self, path):
        """Return layout for an intermediate tensor based on the layout map.

        Allows specifying sharding for layer outputs by including
        patterns like ``"dense.*output"`` in the layout map.

        Args:
            path: String identifying the tensor.

        Returns:
            A ``TensorLayout`` instance, or ``None``.
        """
        if self._layout_map is not None:
            return self._layout_map[path]
        return None


# ---------------------------------------------------------------------------
# Global distribution state (thread-local)
# ---------------------------------------------------------------------------

_DISTRIBUTION_STATE = threading.local()


def distribution():
    """Get the current global distribution instance.

    Returns:
        The current distribution, or ``None``.
    """
    return getattr(_DISTRIBUTION_STATE, "distribution", None)


def set_distribution(value):
    """Set the global distribution.

    Args:
        value: A distribution instance (``DataParallel`` or
            ``ModelParallel``), or ``None``.
    """
    _DISTRIBUTION_STATE.distribution = value


# ---------------------------------------------------------------------------
# Backend bridge: DeviceMesh → dtensor.Mesh
# ---------------------------------------------------------------------------


def _to_backend_mesh(device_mesh):
    """Convert a ``DeviceMesh`` to a ``tf.experimental.dtensor.Mesh``.

    Creates a dtensor mesh with the same shape and axis names.

    Args:
        device_mesh: A ``DeviceMesh`` instance.

    Returns:
        A ``tf.experimental.dtensor.Mesh``, or ``None`` if dtensor is
        unavailable.
    """
    if _dtensor is None:
        return None
    mesh_dims = list(zip(device_mesh.axis_names, device_mesh.shape))
    flat_devices = device_mesh.devices.flatten().tolist()
    try:
        return _dtensor.create_mesh(
            mesh_dims=mesh_dims,
            devices=flat_devices,
        )
    except Exception:
        # Fallback: try create_distributed_mesh for multi-client setups
        try:
            return _dtensor.create_distributed_mesh(
                mesh_dims=mesh_dims,
                local_devices=flat_devices,
            )
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Backend bridge: TensorLayout → dtensor.Layout
# ---------------------------------------------------------------------------


def _to_backend_layout(tensor_layout):
    """Convert a ``TensorLayout`` to a ``tf.experimental.dtensor.Layout``.

    Args:
        tensor_layout: A ``TensorLayout`` instance with a device mesh set.

    Returns:
        A ``tf.experimental.dtensor.Layout``, or ``None`` if dtensor is
        unavailable or mesh is not set.
    """
    if _dtensor is None:
        return None
    if tensor_layout.device_mesh is None:
        return None

    backend_mesh = tensor_layout.device_mesh.backend_mesh
    if backend_mesh is None:
        return None

    # Convert axes: None → UNSHARDED in dtensor
    sharding_specs = []
    for axis in tensor_layout.axes:
        if axis is None:
            sharding_specs.append(_dtensor.UNSHARDED)
        else:
            sharding_specs.append(axis)

    return _dtensor.Layout(sharding_specs, backend_mesh)


# ---------------------------------------------------------------------------
# distribute_tensor
# ---------------------------------------------------------------------------


def distribute_tensor(tensor, layout):
    """Distribute a tensor according to the given layout.

    If dtensor is available, this performs a relayout operation to
    ensure the tensor is sharded according to the specified layout.
    Otherwise, the tensor is returned unchanged.

    Args:
        tensor: A ``tf.Tensor`` to distribute.
        layout: A ``TensorLayout`` or ``tf.experimental.dtensor.Layout``
            describing the desired distribution.

    Returns:
        The distributed tensor. If dtensor is unavailable, returns
        the input tensor unchanged.
    """
    if tensor is None or layout is None:
        return tensor

    if _dtensor is None:
        return tensor

    # Get the backend layout
    if isinstance(layout, TensorLayout):
        backend_layout = layout.backend_layout
    else:
        backend_layout = layout

    if backend_layout is None:
        return tensor

    try:
        return _dtensor.relayout(tensor, backend_layout)
    except Exception:
        # If relayout fails (e.g., tensor not on dtensor mesh),
        # try copy_to_mesh
        try:
            return _dtensor.copy_to_mesh(tensor, backend_layout)
        except Exception:
            return tensor


# ---------------------------------------------------------------------------
# distribute_variable
# ---------------------------------------------------------------------------


def distribute_variable(initial_value, layout, **kwargs):
    """Create a distributed variable with the given layout.

    If dtensor is available, creates a ``DVariable`` with the specified
    layout. Otherwise, creates a standard ``tf.Variable``.

    Args:
        initial_value: Initial value for the variable. Can be a tensor,
            numpy array, or callable (initializer).
        layout: A ``TensorLayout`` or ``tf.experimental.dtensor.Layout``
            specifying how the variable should be distributed.
        **kwargs: Additional keyword arguments passed to the variable
            constructor (e.g., ``name``, ``trainable``, ``dtype``).

    Returns:
        A ``tf.Variable`` or ``dtensor.DVariable``.
    """
    if _dtensor is None or layout is None:
        return tf.Variable(initial_value, **kwargs)

    # Get the backend layout
    if isinstance(layout, TensorLayout):
        backend_layout = layout.backend_layout
    else:
        backend_layout = layout

    if backend_layout is None:
        return tf.Variable(initial_value, **kwargs)

    try:
        # If initial_value is a callable (initializer), call it with layout.
        # Note: shape and dtype are consumed here since DVariable doesn't
        # accept them — they are only needed for the initializer call.
        if callable(initial_value):
            shape = kwargs.pop("shape", None)
            dtype = kwargs.pop("dtype", tf.float32)
            if shape is not None:
                init_val = _dtensor.call_with_layout(
                    initial_value, backend_layout, shape=shape, dtype=dtype
                )
            else:
                init_val = initial_value()
                init_val = _dtensor.copy_to_mesh(init_val, backend_layout)
        else:
            dtype = kwargs.pop("dtype", tf.float32)
            init_tensor = tf.cast(
                tf.convert_to_tensor(initial_value), dtype
            )
            init_val = _dtensor.copy_to_mesh(init_tensor, backend_layout)

        return _dtensor.DVariable(init_val, **kwargs)
    except Exception:
        return tf.Variable(initial_value, **kwargs)


# ---------------------------------------------------------------------------
# distribute_data_input
# ---------------------------------------------------------------------------


def distribute_data_input(data, layout):
    """Distribute input data according to the given layout.

    Packs the input data tensor into a DTensor with the specified layout
    for feeding into a distributed model during training or evaluation.

    For batch-sharded data, the input should be the full global batch;
    this function will handle the sharding according to the layout.

    Args:
        data: A ``tf.Tensor`` or numpy array representing the input data.
        layout: A ``TensorLayout`` or ``tf.experimental.dtensor.Layout``
            specifying how the data should be distributed.

    Returns:
        A distributed tensor. If dtensor is unavailable, returns the
        input data as a regular tensor.
    """
    if data is None or layout is None:
        return data

    if _dtensor is None:
        if not isinstance(data, tf.Tensor):
            return tf.convert_to_tensor(data)
        return data

    if isinstance(layout, TensorLayout):
        backend_layout = layout.backend_layout
    else:
        backend_layout = layout

    if backend_layout is None:
        if not isinstance(data, tf.Tensor):
            return tf.convert_to_tensor(data)
        return data

    try:
        if not isinstance(data, tf.Tensor):
            data = tf.convert_to_tensor(data)

        # Use pack to distribute the data according to layout
        # For a single-client setup, we split data and pack
        mesh = backend_layout.mesh
        batch_dim = None
        for i, spec in enumerate(backend_layout.sharding_specs):
            if spec != _dtensor.UNSHARDED:
                batch_dim = i
                break

        if batch_dim is not None:
            # Get the number of devices along the sharded axis
            shard_axis_name = backend_layout.sharding_specs[batch_dim]
            num_shards = mesh.dim_size(shard_axis_name)
            # Validate batch size is evenly divisible
            batch_size = data.shape[batch_dim]
            if batch_size is not None and batch_size % num_shards != 0:
                raise ValueError(
                    f"Data dimension {batch_dim} (size={batch_size}) is not "
                    f"evenly divisible by the number of shards "
                    f"({num_shards}) along axis '{shard_axis_name}'."
                )
            # Split data along the batch dimension
            splits = tf.split(data, num_shards, axis=batch_dim)
            return _dtensor.pack(splits, backend_layout)
        else:
            # Fully replicated — pack identical copies to each device.
            # Note: this path is for single-client setups. Multi-client
            # distributed training may require different logic.
            num_devices = mesh.num_local_devices()
            copies = [data] * num_devices
            return _dtensor.pack(copies, backend_layout)
    except Exception:
        # Fallback: try relayout
        try:
            return _dtensor.relayout(
                tf.convert_to_tensor(data) if not isinstance(data, tf.Tensor) else data,
                backend_layout,
            )
        except Exception:
            if not isinstance(data, tf.Tensor):
                return tf.convert_to_tensor(data)
            return data


# ---------------------------------------------------------------------------
# Utility: get variable path for layout map lookup
# ---------------------------------------------------------------------------


def _get_variable_path(variable):
    """Extract a path string from a variable for layout map matching.

    Attempts to derive a path like ``"layer_name.weight_name"`` from
    the variable's name.

    Args:
        variable: A ``tf.Variable`` or similar.

    Returns:
        A string path for regex matching in the layout map.
    """
    name = getattr(variable, "name", "")
    # tf.Variable names typically look like "layer_name/weight_name:0"
    # Strip the ":0" suffix
    if ":" in name:
        name = name.rsplit(":", 1)[0]
    # Replace "/" with "." for Keras 3 style path matching
    # but also keep "/" for Keras 2 style matching
    return name
