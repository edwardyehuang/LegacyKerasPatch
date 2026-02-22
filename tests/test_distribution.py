"""Tests for the distribution module."""

import pytest
import collections.abc


class TestDistributionModule:
    """Test cases for the distribution module imports and structure."""

    def test_import(self):
        """Test that the distribution module can be imported."""
        from legacy_keras_patch import distribution
        assert distribution is not None

    def test_layout_map_class_exists(self):
        """Test that LayoutMap class is available."""
        from legacy_keras_patch.distribution import LayoutMap
        assert LayoutMap is not None

    def test_layout_map_is_mutable_mapping(self):
        """Test that LayoutMap implements MutableMapping."""
        from legacy_keras_patch.distribution import LayoutMap
        assert issubclass(LayoutMap, collections.abc.MutableMapping)

    def test_device_mesh_class_exists(self):
        """Test that DeviceMesh class is available."""
        from legacy_keras_patch.distribution import DeviceMesh
        assert DeviceMesh is not None

    def test_tensor_layout_class_exists(self):
        """Test that TensorLayout class is available."""
        from legacy_keras_patch.distribution import TensorLayout
        assert TensorLayout is not None

    def test_data_parallel_class_exists(self):
        """Test that DataParallel class is available."""
        from legacy_keras_patch.distribution import DataParallel
        assert DataParallel is not None

    def test_model_parallel_class_exists(self):
        """Test that ModelParallel class is available."""
        from legacy_keras_patch.distribution import ModelParallel
        assert ModelParallel is not None

    def test_list_devices(self):
        """Test list_devices returns a list."""
        from legacy_keras_patch.distribution import list_devices
        devices = list_devices()
        assert isinstance(devices, list)
        assert len(devices) > 0

    def test_distribution_default_none(self):
        """Test that distribution() returns None by default."""
        from legacy_keras_patch.distribution import distribution
        result = distribution()
        assert result is None

    def test_set_distribution(self):
        """Test set_distribution and distribution getter."""
        from legacy_keras_patch.distribution import (
            set_distribution, distribution, DataParallel
        )
        dp = DataParallel()
        set_distribution(dp)
        assert distribution() is dp
        # Clean up
        set_distribution(None)
        assert distribution() is None

    def test_distribute_tensor(self):
        """Test distribute_tensor returns tensor unchanged."""
        import tensorflow as tf
        from legacy_keras_patch.distribution import distribute_tensor
        tensor = tf.constant([1.0, 2.0, 3.0])
        result = distribute_tensor(tensor, None)
        assert result is tensor

    def test_all_exports(self):
        """Test __all__ contains expected exports."""
        from legacy_keras_patch.distribution import __all__
        expected = [
            'DataParallel', 'DeviceMesh', 'LayoutMap', 'ModelParallel',
            'TensorLayout', 'distribute_tensor', 'distribution',
            'list_devices', 'set_distribution',
        ]
        for item in expected:
            assert item in __all__


class TestDeviceMesh:
    """Test cases for DeviceMesh."""

    def test_creation(self):
        """Test DeviceMesh creation with explicit devices."""
        from legacy_keras_patch.distribution import DeviceMesh
        mesh = DeviceMesh(
            shape=(2,),
            axis_names=["batch"],
            devices=["cpu:0", "cpu:1"],
        )
        assert mesh.shape == (2,)
        assert mesh.axis_names == ["batch"]

    def test_empty_shape_raises(self):
        """Test that empty shape raises ValueError."""
        from legacy_keras_patch.distribution import DeviceMesh
        with pytest.raises(ValueError):
            DeviceMesh(shape=(), axis_names=[])

    def test_mismatched_shape_axis_names_raises(self):
        """Test that mismatched shape and axis_names raises ValueError."""
        from legacy_keras_patch.distribution import DeviceMesh
        with pytest.raises(ValueError):
            DeviceMesh(shape=(2, 2), axis_names=["batch"], devices=["a", "b", "c", "d"])

    def test_mismatched_devices_raises(self):
        """Test that device count mismatch raises ValueError."""
        from legacy_keras_patch.distribution import DeviceMesh
        with pytest.raises(ValueError):
            DeviceMesh(shape=(4,), axis_names=["batch"], devices=["cpu:0", "cpu:1"])

    def test_repr(self):
        """Test DeviceMesh repr."""
        from legacy_keras_patch.distribution import DeviceMesh
        mesh = DeviceMesh(shape=(2,), axis_names=["batch"], devices=["a", "b"])
        r = repr(mesh)
        assert "DeviceMesh" in r
        assert "batch" in r


class TestTensorLayout:
    """Test cases for TensorLayout."""

    def test_creation(self):
        """Test TensorLayout creation."""
        from legacy_keras_patch.distribution import TensorLayout
        layout = TensorLayout(axes=(None, "model"))
        assert layout.axes == (None, "model")
        assert layout.device_mesh is None

    def test_creation_with_mesh(self):
        """Test TensorLayout creation with device_mesh."""
        from legacy_keras_patch.distribution import TensorLayout, DeviceMesh
        mesh = DeviceMesh(shape=(2,), axis_names=["model"], devices=["a", "b"])
        layout = TensorLayout(axes=(None, "model"), device_mesh=mesh)
        assert layout.device_mesh is mesh

    def test_invalid_axes_raises(self):
        """Test that invalid axes raise ValueError."""
        from legacy_keras_patch.distribution import TensorLayout, DeviceMesh
        mesh = DeviceMesh(shape=(2,), axis_names=["model"], devices=["a", "b"])
        with pytest.raises(ValueError):
            TensorLayout(axes=("invalid_axis",), device_mesh=mesh)

    def test_device_mesh_setter(self):
        """Test setting device_mesh after creation."""
        from legacy_keras_patch.distribution import TensorLayout, DeviceMesh
        layout = TensorLayout(axes=(None,))
        mesh = DeviceMesh(shape=(2,), axis_names=["model"], devices=["a", "b"])
        layout.device_mesh = mesh
        assert layout.device_mesh is mesh

    def test_device_mesh_override_raises(self):
        """Test that overriding device_mesh raises ValueError."""
        from legacy_keras_patch.distribution import TensorLayout, DeviceMesh
        mesh1 = DeviceMesh(shape=(2,), axis_names=["model"], devices=["a", "b"])
        mesh2 = DeviceMesh(shape=(2,), axis_names=["batch"], devices=["c", "d"])
        layout = TensorLayout(axes=(None,), device_mesh=mesh1)
        with pytest.raises(ValueError):
            layout.device_mesh = mesh2

    def test_repr(self):
        """Test TensorLayout repr."""
        from legacy_keras_patch.distribution import TensorLayout
        layout = TensorLayout(axes=(None, "model"))
        r = repr(layout)
        assert "TensorLayout" in r


class TestLayoutMapFallback:
    """Test cases for the fallback LayoutMap (when Keras 2 dtensor unavailable).

    These tests exercise the standalone LayoutMap implementation. When
    the Keras 2 dtensor LayoutMap is available, some behaviour details
    (e.g. value types) may differ, so we test core dict-like behaviour.
    """

    def test_layout_map_len(self):
        """Test LayoutMap length tracking."""
        from legacy_keras_patch.distribution import LayoutMap
        lm = LayoutMap(mesh=None)
        assert len(lm) == 0

    def test_layout_map_is_mapping(self):
        """Test that LayoutMap is a MutableMapping."""
        from legacy_keras_patch.distribution import LayoutMap
        assert issubclass(LayoutMap, collections.abc.MutableMapping)


class TestDataParallel:
    """Test cases for DataParallel."""

    def test_creation_defaults(self):
        """Test DataParallel with defaults."""
        from legacy_keras_patch.distribution import DataParallel
        dp = DataParallel()
        assert dp.device_mesh is None
        assert dp.auto_shard_dataset is True

    def test_creation_with_mesh(self):
        """Test DataParallel with mesh."""
        from legacy_keras_patch.distribution import DataParallel, DeviceMesh
        mesh = DeviceMesh(shape=(2,), axis_names=["batch"], devices=["a", "b"])
        dp = DataParallel(device_mesh=mesh)
        assert dp.device_mesh is mesh


class TestModelParallel:
    """Test cases for ModelParallel."""

    def test_creation_defaults(self):
        """Test ModelParallel with defaults."""
        from legacy_keras_patch.distribution import ModelParallel
        mp = ModelParallel()
        assert mp.device_mesh is None
        assert mp.layout_map is None
        assert mp.batch_dim_name is None

    def test_creation_with_args(self):
        """Test ModelParallel with arguments."""
        from legacy_keras_patch.distribution import (
            ModelParallel, DeviceMesh, LayoutMap
        )
        mesh = DeviceMesh(shape=(2,), axis_names=["model"], devices=["a", "b"])
        lm = LayoutMap(mesh=None)
        mp = ModelParallel(device_mesh=mesh, layout_map=lm, batch_dim_name="batch")
        assert mp.device_mesh is mesh
        assert mp.layout_map is lm
        assert mp.batch_dim_name == "batch"


class TestMainModuleDistribution:
    """Test that distribution is part of the main module exports."""

    def test_distribution_in_all(self):
        """Test distribution is in __all__."""
        from legacy_keras_patch import __all__
        assert 'distribution' in __all__

    def test_distribution_import_from_main(self):
        """Test distribution can be imported from main module."""
        from legacy_keras_patch import distribution
        assert hasattr(distribution, 'LayoutMap')
