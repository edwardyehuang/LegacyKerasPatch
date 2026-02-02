"""Tests for the main legacy_keras_patch module."""

import pytest


class TestMainModule:
    """Test cases for main module functions."""
    
    def test_import(self):
        """Test that the module can be imported."""
        import legacy_keras_patch
        assert legacy_keras_patch is not None
    
    def test_version(self):
        """Test that version is defined."""
        from legacy_keras_patch import __version__
        assert __version__ == "1.0.0"
    
    def test_ops_import(self):
        """Test that ops module can be imported."""
        from legacy_keras_patch import ops
        assert ops is not None
    
    def test_get_keras_version(self):
        """Test get_keras_version function."""
        from legacy_keras_patch import get_keras_version
        version = get_keras_version()
        # Should return an integer (2 or 3) or None
        assert version is None or isinstance(version, int)
        if version is not None:
            assert version >= 2  # Keras 2 or higher
    
    def test_is_keras_3(self):
        """Test is_keras_3 function."""
        from legacy_keras_patch import is_keras_3, get_keras_version
        result = is_keras_3()
        assert isinstance(result, bool)
        # Verify consistency with get_keras_version
        version = get_keras_version()
        if version is not None:
            assert result == (version >= 3)
    
    def test_apply_patch(self):
        """Test apply_patch function."""
        from legacy_keras_patch import apply_patch
        # Should not raise an exception
        apply_patch()
    
    def test_is_patched_after_apply(self):
        """Test is_patched returns True after applying patch."""
        from legacy_keras_patch import apply_patch, is_patched
        apply_patch()
        # After apply_patch, keras.ops should be available
        result = is_patched()
        assert isinstance(result, bool)
        # With Keras 3, keras.ops exists natively, with Keras 2 it should be patched
        assert result is True
    
    def test_public_api_exports(self):
        """Test that all public API functions are exported."""
        import legacy_keras_patch
        assert hasattr(legacy_keras_patch, 'apply_patch')
        assert hasattr(legacy_keras_patch, 'is_patched')
        assert hasattr(legacy_keras_patch, 'get_keras_version')
        assert hasattr(legacy_keras_patch, 'is_keras_3')
        assert hasattr(legacy_keras_patch, 'ops')
    
    def test_all_exports(self):
        """Test __all__ contains expected exports."""
        from legacy_keras_patch import __all__
        expected = ['apply_patch', 'is_patched', 'get_keras_version', 'is_keras_3', 'ops']
        for item in expected:
            assert item in __all__


class TestApplyPatchIdempotent:
    """Test that apply_patch is idempotent."""
    
    def test_multiple_calls(self):
        """Test that calling apply_patch multiple times is safe."""
        from legacy_keras_patch import apply_patch
        # Call multiple times should not raise
        apply_patch()
        apply_patch()
        apply_patch()


class TestOpsSubmodules:
    """Test that ops submodules are accessible."""
    
    def test_nn_submodule(self):
        """Test that ops.nn is accessible."""
        from legacy_keras_patch.ops import nn
        assert nn is not None
    
    def test_image_submodule(self):
        """Test that ops.image is accessible."""
        from legacy_keras_patch.ops import image
        assert image is not None
    
    def test_linalg_submodule(self):
        """Test that ops.linalg is accessible."""
        from legacy_keras_patch.ops import linalg
        assert linalg is not None
    
    def test_numpy_submodule(self):
        """Test that ops.numpy is accessible."""
        from legacy_keras_patch.ops import numpy
        assert numpy is not None
