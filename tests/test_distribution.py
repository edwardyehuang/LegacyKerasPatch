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

    def test_distribution_base_class_exists(self):
        """Test that Distribution base class is available."""
        from legacy_keras_patch.distribution import Distribution
        assert Distribution is not None

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

    def test_distribute_tensor_none_layout(self):
        """Test distribute_tensor returns tensor unchanged with None layout."""
        import tensorflow as tf
        from legacy_keras_patch.distribution import distribute_tensor
        tensor = tf.constant([1.0, 2.0, 3.0])
        result = distribute_tensor(tensor, None)
        assert result is tensor

    def test_distribute_tensor_none_tensor(self):
        """Test distribute_tensor returns None for None tensor."""
        from legacy_keras_patch.distribution import distribute_tensor
        result = distribute_tensor(None, None)
        assert result is None

    def test_all_exports(self):
        """Test __all__ contains expected exports."""
        from legacy_keras_patch.distribution import __all__
        expected = [
            'DataParallel', 'DeviceMesh', 'Distribution', 'LayoutMap',
            'ModelParallel', 'TensorLayout', 'distribute_data_input',
            'distribute_tensor', 'distribute_variable', 'distribution',
            'list_devices', 'set_distribution',
        ]
        for item in expected:
            assert item in __all__


class TestDeviceMesh:
    """Test cases for DeviceMesh."""

    def test_creation_1d(self):
        """Test DeviceMesh creation with 1D shape."""
        from legacy_keras_patch.distribution import DeviceMesh
        mesh = DeviceMesh(
            shape=(2,),
            axis_names=["batch"],
            devices=["cpu:0", "cpu:1"],
        )
        assert mesh.shape == (2,)
        assert mesh.axis_names == ["batch"]

    def test_creation_2d(self):
        """Test DeviceMesh creation with 2D shape."""
        from legacy_keras_patch.distribution import DeviceMesh
        mesh = DeviceMesh(
            shape=(2, 2),
            axis_names=["batch", "model"],
            devices=["cpu:0", "cpu:1", "cpu:2", "cpu:3"],
        )
        assert mesh.shape == (2, 2)
        assert mesh.axis_names == ["batch", "model"]
        assert mesh.devices.shape == (2, 2)

    def test_creation_3d(self):
        """Test DeviceMesh creation with 3D shape."""
        from legacy_keras_patch.distribution import DeviceMesh
        devices = [f"cpu:{i}" for i in range(8)]
        mesh = DeviceMesh(
            shape=(2, 2, 2),
            axis_names=["batch", "model", "expert"],
            devices=devices,
        )
        assert mesh.shape == (2, 2, 2)
        assert mesh.axis_names == ["batch", "model", "expert"]
        assert mesh.devices.shape == (2, 2, 2)

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

    def test_2d_repr(self):
        """Test DeviceMesh repr for 2D mesh."""
        from legacy_keras_patch.distribution import DeviceMesh
        mesh = DeviceMesh(
            shape=(2, 2),
            axis_names=["batch", "model"],
            devices=["a", "b", "c", "d"],
        )
        r = repr(mesh)
        assert "batch" in r
        assert "model" in r
        assert "(2, 2)" in r


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

    def test_creation_2d_mesh(self):
        """Test TensorLayout with 2D mesh axes."""
        from legacy_keras_patch.distribution import TensorLayout, DeviceMesh
        mesh = DeviceMesh(
            shape=(2, 2), axis_names=["batch", "model"],
            devices=["a", "b", "c", "d"],
        )
        layout = TensorLayout(axes=("batch", "model"), device_mesh=mesh)
        assert layout.axes == ("batch", "model")
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

    def test_is_fully_replicated(self):
        """Test is_fully_replicated property."""
        from legacy_keras_patch.distribution import TensorLayout
        layout_rep = TensorLayout(axes=(None, None))
        assert layout_rep.is_fully_replicated is True

        layout_sharded = TensorLayout(axes=(None, "model"))
        assert layout_sharded.is_fully_replicated is False

    def test_equality(self):
        """Test TensorLayout equality."""
        from legacy_keras_patch.distribution import TensorLayout, DeviceMesh
        mesh = DeviceMesh(shape=(2,), axis_names=["model"], devices=["a", "b"])
        l1 = TensorLayout(axes=(None, "model"), device_mesh=mesh)
        l2 = TensorLayout(axes=(None, "model"), device_mesh=mesh)
        assert l1 == l2

    def test_inequality(self):
        """Test TensorLayout inequality."""
        from legacy_keras_patch.distribution import TensorLayout
        l1 = TensorLayout(axes=(None, "model"))
        l2 = TensorLayout(axes=("model", None))
        assert l1 != l2

    def test_repr(self):
        """Test TensorLayout repr."""
        from legacy_keras_patch.distribution import TensorLayout
        layout = TensorLayout(axes=(None, "model"))
        r = repr(layout)
        assert "TensorLayout" in r


class TestLayoutMap:
    """Test cases for LayoutMap."""

    def test_layout_map_len(self):
        """Test LayoutMap length tracking."""
        from legacy_keras_patch.distribution import LayoutMap
        lm = LayoutMap(mesh=None)
        assert len(lm) == 0

    def test_layout_map_is_mapping(self):
        """Test that LayoutMap is a MutableMapping."""
        from legacy_keras_patch.distribution import LayoutMap
        assert issubclass(LayoutMap, collections.abc.MutableMapping)

    def test_layout_map_set_tuple(self):
        """Test setting a tuple value in LayoutMap."""
        from legacy_keras_patch.distribution import LayoutMap, TensorLayout
        lm = LayoutMap()
        lm['dense.*kernel'] = (None, "model")
        result = lm['dense.*kernel']
        assert isinstance(result, TensorLayout)
        assert result.axes == (None, "model")

    def test_layout_map_regex_lookup(self):
        """Test regex-based lookup in LayoutMap."""
        from legacy_keras_patch.distribution import LayoutMap, DeviceMesh
        mesh = DeviceMesh(shape=(2,), axis_names=["model"], devices=["a", "b"])
        lm = LayoutMap(device_mesh=mesh)
        lm['dense.*kernel'] = (None, "model")
        # Should match via regex
        result = lm['dense_1/kernel']
        assert result is not None
        assert result.axes == (None, "model")

    def test_layout_map_no_match(self):
        """Test LayoutMap returns None for non-matching key."""
        from legacy_keras_patch.distribution import LayoutMap
        lm = LayoutMap()
        lm['dense.*kernel'] = (None, "model")
        result = lm['conv2d_1/kernel']
        assert result is None

    def test_layout_map_multiple_match_raises(self):
        """Test that multiple regex matches raise ValueError."""
        from legacy_keras_patch.distribution import LayoutMap
        lm = LayoutMap()
        lm['.*kernel'] = (None, "model")
        lm['dense.*'] = ("model", None)
        with pytest.raises(ValueError):
            lm['dense_1/kernel']

    def test_layout_map_duplicate_key_raises(self):
        """Test that duplicate keys raise ValueError."""
        from legacy_keras_patch.distribution import LayoutMap
        lm = LayoutMap()
        lm['dense.*kernel'] = (None, "model")
        with pytest.raises(ValueError):
            lm['dense.*kernel'] = ("model", None)

    def test_layout_map_delete(self):
        """Test deleting from LayoutMap."""
        from legacy_keras_patch.distribution import LayoutMap
        lm = LayoutMap()
        lm['dense.*kernel'] = (None, "model")
        assert len(lm) == 1
        del lm['dense.*kernel']
        assert len(lm) == 0

    def test_layout_map_populates_device_mesh(self):
        """Test that LayoutMap populates device_mesh on layouts."""
        from legacy_keras_patch.distribution import LayoutMap, DeviceMesh
        mesh = DeviceMesh(shape=(2,), axis_names=["model"], devices=["a", "b"])
        lm = LayoutMap(device_mesh=mesh)
        lm['dense.*kernel'] = (None, "model")
        result = lm['dense.*kernel']
        assert result.device_mesh is mesh

    def test_layout_map_2d_mesh(self):
        """Test LayoutMap with 2D mesh patterns."""
        from legacy_keras_patch.distribution import LayoutMap, DeviceMesh
        mesh = DeviceMesh(
            shape=(2, 2), axis_names=["batch", "model"],
            devices=["a", "b", "c", "d"],
        )
        lm = LayoutMap(device_mesh=mesh)
        lm['.*kernel'] = (None, "model")
        lm['.*bias'] = ("model",)
        # Test lookups
        kernel_layout = lm['dense_1/kernel']
        assert kernel_layout.axes == (None, "model")
        bias_layout = lm['dense_1/bias']
        assert bias_layout.axes == ("model",)


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
        assert dp.batch_dim_name == "batch"

    def test_creation_with_devices(self):
        """Test DataParallel with devices list."""
        from legacy_keras_patch.distribution import DataParallel
        dp = DataParallel(devices=["cpu:0", "cpu:1"])
        assert dp.device_mesh is not None
        assert dp.device_mesh.shape == (2,)

    def test_get_data_layout(self):
        """Test DataParallel.get_data_layout shards batch dim."""
        from legacy_keras_patch.distribution import DataParallel, DeviceMesh
        mesh = DeviceMesh(shape=(4,), axis_names=["batch"],
                         devices=["a", "b", "c", "d"])
        dp = DataParallel(device_mesh=mesh)
        layout = dp.get_data_layout((32, 224, 224, 3))
        assert layout.axes == ("batch", None, None, None)

    def test_get_variable_layout_replicated(self):
        """Test DataParallel.get_variable_layout returns replicated."""
        import tensorflow as tf
        from legacy_keras_patch.distribution import DataParallel, DeviceMesh
        mesh = DeviceMesh(shape=(4,), axis_names=["batch"],
                         devices=["a", "b", "c", "d"])
        dp = DataParallel(device_mesh=mesh)
        var = tf.Variable(tf.zeros([10, 5]), name="test_var")
        layout = dp.get_variable_layout(var)
        assert layout.axes == (None, None)
        assert layout.is_fully_replicated

    def test_num_model_replicas(self):
        """Test DataParallel.num_model_replicas equals device count."""
        from legacy_keras_patch.distribution import DataParallel, DeviceMesh
        mesh = DeviceMesh(shape=(4,), axis_names=["batch"],
                         devices=["a", "b", "c", "d"])
        dp = DataParallel(device_mesh=mesh)
        assert dp.num_model_replicas == 4

    def test_scope_context_manager(self):
        """Test DataParallel scope context manager."""
        from legacy_keras_patch.distribution import (
            DataParallel, DeviceMesh, distribution, set_distribution
        )
        mesh = DeviceMesh(shape=(2,), axis_names=["batch"], devices=["a", "b"])
        dp = DataParallel(device_mesh=mesh)
        assert distribution() is None
        with dp.scope():
            assert distribution() is dp
        assert distribution() is None


class TestModelParallel:
    """Test cases for ModelParallel."""

    def test_creation_defaults(self):
        """Test ModelParallel with defaults."""
        from legacy_keras_patch.distribution import ModelParallel
        mp = ModelParallel()
        assert mp.device_mesh is None
        assert mp.layout_map is None
        assert mp.batch_dim_name is None

    def test_creation_with_layout_map(self):
        """Test ModelParallel infers device_mesh from layout_map."""
        from legacy_keras_patch.distribution import (
            ModelParallel, DeviceMesh, LayoutMap
        )
        mesh = DeviceMesh(
            shape=(2, 2), axis_names=["batch", "model"],
            devices=["a", "b", "c", "d"],
        )
        lm = LayoutMap(device_mesh=mesh)
        mp = ModelParallel(layout_map=lm)
        assert mp.device_mesh is mesh
        assert mp.batch_dim_name == "batch"

    def test_creation_with_explicit_batch_dim(self):
        """Test ModelParallel with explicit batch_dim_name."""
        from legacy_keras_patch.distribution import (
            ModelParallel, DeviceMesh, LayoutMap
        )
        mesh = DeviceMesh(
            shape=(2, 2), axis_names=["data", "model"],
            devices=["a", "b", "c", "d"],
        )
        lm = LayoutMap(device_mesh=mesh)
        mp = ModelParallel(layout_map=lm, batch_dim_name="data")
        assert mp.batch_dim_name == "data"

    def test_get_data_layout_2d_mesh(self):
        """Test ModelParallel.get_data_layout with 2D mesh."""
        from legacy_keras_patch.distribution import (
            ModelParallel, DeviceMesh, LayoutMap
        )
        mesh = DeviceMesh(
            shape=(4, 2), axis_names=["batch", "model"],
            devices=[f"d:{i}" for i in range(8)],
        )
        lm = LayoutMap(device_mesh=mesh)
        mp = ModelParallel(layout_map=lm, batch_dim_name="batch")
        layout = mp.get_data_layout((32, 128))
        assert layout.axes == ("batch", None)

    def test_get_variable_layout_from_map(self):
        """Test ModelParallel.get_variable_layout uses layout map."""
        import tensorflow as tf
        from legacy_keras_patch.distribution import (
            ModelParallel, DeviceMesh, LayoutMap
        )
        mesh = DeviceMesh(
            shape=(2, 2), axis_names=["batch", "model"],
            devices=["a", "b", "c", "d"],
        )
        lm = LayoutMap(device_mesh=mesh)
        lm['.*kernel'] = (None, "model")
        mp = ModelParallel(layout_map=lm)
        var = tf.Variable(tf.zeros([10, 5]), name="dense/kernel:0")
        layout = mp.get_variable_layout(var)
        assert layout.axes == (None, "model")

    def test_get_variable_layout_unmatched_replicated(self):
        """Test ModelParallel returns replicated for unmatched variables."""
        import tensorflow as tf
        from legacy_keras_patch.distribution import (
            ModelParallel, DeviceMesh, LayoutMap
        )
        mesh = DeviceMesh(
            shape=(2, 2), axis_names=["batch", "model"],
            devices=["a", "b", "c", "d"],
        )
        lm = LayoutMap(device_mesh=mesh)
        lm['.*kernel'] = (None, "model")
        mp = ModelParallel(layout_map=lm)
        var = tf.Variable(tf.zeros([10]), name="batch_norm/gamma:0")
        layout = mp.get_variable_layout(var)
        assert layout.axes == (None,)
        assert layout.is_fully_replicated

    def test_get_tensor_layout(self):
        """Test ModelParallel.get_tensor_layout uses layout map."""
        from legacy_keras_patch.distribution import (
            ModelParallel, DeviceMesh, LayoutMap
        )
        mesh = DeviceMesh(
            shape=(2, 2), axis_names=["batch", "model"],
            devices=["a", "b", "c", "d"],
        )
        lm = LayoutMap(device_mesh=mesh)
        lm['.*dense.*output'] = ("batch", None)
        mp = ModelParallel(layout_map=lm)
        layout = mp.get_tensor_layout("dense_1/output")
        assert layout is not None
        assert layout.axes == ("batch", None)

    def test_num_model_replicas_2d_mesh(self):
        """Test ModelParallel.num_model_replicas with 2D mesh."""
        from legacy_keras_patch.distribution import (
            ModelParallel, DeviceMesh, LayoutMap
        )
        mesh = DeviceMesh(
            shape=(4, 2), axis_names=["batch", "model"],
            devices=[f"d:{i}" for i in range(8)],
        )
        lm = LayoutMap(device_mesh=mesh)
        mp = ModelParallel(layout_map=lm, batch_dim_name="batch")
        # batch axis has size 4 → 4 model replicas
        assert mp.num_model_replicas == 4

    def test_scope_context_manager(self):
        """Test ModelParallel scope context manager."""
        from legacy_keras_patch.distribution import (
            ModelParallel, DeviceMesh, LayoutMap, distribution
        )
        mesh = DeviceMesh(
            shape=(2, 2), axis_names=["batch", "model"],
            devices=["a", "b", "c", "d"],
        )
        lm = LayoutMap(device_mesh=mesh)
        mp = ModelParallel(layout_map=lm)
        assert distribution() is None
        with mp.scope():
            assert distribution() is mp
        assert distribution() is None

    def test_scope_nesting(self):
        """Test nested scope restores previous distribution."""
        from legacy_keras_patch.distribution import (
            ModelParallel, DataParallel, DeviceMesh, LayoutMap, distribution
        )
        mesh = DeviceMesh(shape=(2,), axis_names=["batch"], devices=["a", "b"])
        dp = DataParallel(device_mesh=mesh)

        mesh2 = DeviceMesh(
            shape=(2, 2), axis_names=["batch", "model"],
            devices=["a", "b", "c", "d"],
        )
        lm = LayoutMap(device_mesh=mesh2)
        mp = ModelParallel(layout_map=lm)

        with dp.scope():
            assert distribution() is dp
            with mp.scope():
                assert distribution() is mp
            assert distribution() is dp
        assert distribution() is None


class TestDistributeVariable:
    """Test cases for distribute_variable."""

    def test_no_dtensor_returns_tf_variable(self):
        """Test fallback to tf.Variable when dtensor unavailable."""
        import tensorflow as tf
        from legacy_keras_patch.distribution import distribute_variable
        var = distribute_variable(
            tf.zeros([3, 3]),
            layout=None,
            name="test_var",
        )
        assert isinstance(var, tf.Variable)

    def test_with_none_layout(self):
        """Test that None layout creates a regular variable."""
        import tensorflow as tf
        from legacy_keras_patch.distribution import distribute_variable
        var = distribute_variable(tf.ones([2, 2]), layout=None, name="test")
        assert isinstance(var, tf.Variable)


class TestDistributeDataInput:
    """Test cases for distribute_data_input."""

    def test_none_data(self):
        """Test distribute_data_input with None data."""
        from legacy_keras_patch.distribution import distribute_data_input
        result = distribute_data_input(None, None)
        assert result is None

    def test_none_layout_returns_data_unchanged(self):
        """Test distribute_data_input with None layout returns data unchanged."""
        import numpy as np
        from legacy_keras_patch.distribution import distribute_data_input
        data = np.ones((4, 3))
        result = distribute_data_input(data, None)
        # With no layout, data is returned as-is (no unnecessary conversion)
        assert result is data

    def test_tensor_input_none_layout(self):
        """Test distribute_data_input with tensor input and None layout."""
        import tensorflow as tf
        from legacy_keras_patch.distribution import distribute_data_input
        data = tf.constant([1.0, 2.0, 3.0])
        result = distribute_data_input(data, None)
        assert result is data


class TestGetVariablePath:
    """Test cases for _get_variable_path utility."""

    def test_strips_colon_suffix(self):
        """Test that :0 suffix is stripped from variable names."""
        import tensorflow as tf
        from legacy_keras_patch.distribution import _get_variable_path
        var = tf.Variable(tf.zeros([2, 2]), name="dense/kernel")
        path = _get_variable_path(var)
        # Should not contain ":0"
        assert ":0" not in path
        assert "kernel" in path

    def test_preserves_path_structure(self):
        """Test that path structure is preserved."""
        import tensorflow as tf
        from legacy_keras_patch.distribution import _get_variable_path
        var = tf.Variable(tf.zeros([2]), name="model/dense_1/bias")
        path = _get_variable_path(var)
        assert "dense_1" in path
        assert "bias" in path


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
