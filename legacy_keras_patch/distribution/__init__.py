"""
Keras 3 distribution compatibility layer for Keras 2.

This module provides keras.distribution compatible classes and functions,
wrapping Keras 2's ``keras.dtensor`` functionality (specifically
``keras.src.dtensor.layout_map.LayoutMap``) where available, and falling
back to standalone implementations otherwise.
"""

import collections
import collections.abc
import re

import numpy as np
import tensorflow as tf

__all__ = [
    "DataParallel",
    "DeviceMesh",
    "LayoutMap",
    "ModelParallel",
    "TensorLayout",
    "distribute_tensor",
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


class DeviceMesh:
    """A cluster of computation devices for distributed computation.

    This API is aligned with ``tf.experimental.dtensor.Mesh``.

    Args:
        shape: tuple or list of integers. The shape of the ``DeviceMesh``.
        axis_names: List of strings. The logical name of each axis.
        devices: Optional list of devices. Defaults to all available devices.
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
        if np.prod(shape) != np.prod(devices.shape):
            raise ValueError(
                "Shape does not match the number of devices. "
                f"Received: shape={shape}; devices.shape="
                f"{devices.shape}"
            )
        self._shape = tuple(shape)
        self._axis_names = list(axis_names)
        self._devices = np.reshape(devices, shape)

    @property
    def shape(self):
        return self._shape

    @property
    def axis_names(self):
        return self._axis_names

    @property
    def devices(self):
        return self._devices

    def __repr__(self):
        return (
            f"<{self.__class__.__name__} "
            f"shape={self.shape}, axis_names={self.axis_names}>"
        )

    def __str__(self):
        return self.__repr__()


class TensorLayout:
    """A layout to apply to a tensor.

    This API is aligned with ``tf.experimental.dtensor.Layout``.

    Args:
        axes: tuple of strings that should map to the ``axis_names`` in
            a ``DeviceMesh``. Use ``None`` as a placeholder for dimensions
            that don't need sharding.
        device_mesh: Optional ``DeviceMesh`` instance.
    """

    def __init__(self, axes, device_mesh=None):
        self._axes = tuple(axes)
        self._device_mesh = device_mesh
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
        self._validate_axes()

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


# ---------------------------------------------------------------------------
# LayoutMap — wraps keras.src.dtensor.layout_map.LayoutMap when available,
# otherwise falls back to a standalone implementation.
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


try:
    from keras.src.dtensor.layout_map import (
        LayoutMap,  # Keras 2 (internal)  # noqa: F811
    )
except Exception:
    try:
        from keras.dtensor.experimental import (
            LayoutMap,  # Keras 2 (public)  # noqa: F811
        )
    except Exception:
        pass


class DataParallel:
    """Distribution that replicates model across devices.

    Args:
        device_mesh: Optional ``DeviceMesh`` instance.
        auto_shard_dataset: Whether to auto-shard the dataset. Defaults
            to ``True``.
    """

    def __init__(self, device_mesh=None, auto_shard_dataset=True):
        self._device_mesh = device_mesh
        self._auto_shard_dataset = auto_shard_dataset

    @property
    def device_mesh(self):
        return self._device_mesh

    @property
    def auto_shard_dataset(self):
        return self._auto_shard_dataset


class ModelParallel:
    """Distribution that shards model variables according to a layout map.

    Args:
        device_mesh: Optional ``DeviceMesh`` instance.
        layout_map: Optional ``LayoutMap`` instance.
        batch_dim_name: The name of the batch dimension.
    """

    def __init__(self, device_mesh=None, layout_map=None, batch_dim_name=None):
        self._device_mesh = device_mesh
        self._layout_map = layout_map
        self._batch_dim_name = batch_dim_name

    @property
    def device_mesh(self):
        return self._device_mesh

    @property
    def layout_map(self):
        return self._layout_map

    @property
    def batch_dim_name(self):
        return self._batch_dim_name


_global_distribution = None


def distribution():
    """Get the current global distribution instance.

    Returns:
        The current distribution, or ``None``.
    """
    return _global_distribution


def set_distribution(value):
    """Set the global distribution.

    Args:
        value: A distribution instance (``DataParallel`` or
            ``ModelParallel``), or ``None``.
    """
    global _global_distribution
    _global_distribution = value


def distribute_tensor(tensor, layout):
    """Distribute *tensor* according to *layout*.

    This is a no-op shim for Keras 2. The tensor is returned unchanged.

    Args:
        tensor: The tensor to distribute.
        layout: A ``TensorLayout`` describing the desired distribution.

    Returns:
        The tensor (unchanged in Keras 2).
    """
    return tensor
