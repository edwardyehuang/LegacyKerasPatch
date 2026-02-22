"""
LegacyKerasPatch - Keras 3 ops compatibility layer for Keras 2.

This package provides a patch that simulates Keras 3's `keras.ops` module for Keras 2 users,
enabling them to migrate their code to use `keras.ops` in preparation for a future switch to Keras 3.

The patch is a wrapper that forwards operations to corresponding TensorFlow operations.
When Keras 3 is detected, the patch does nothing since keras.ops is already available.

Usage:
    from legacy_keras_patch import apply_patch
    apply_patch()  # Call this before using keras.ops
    
    # Now you can use keras.ops
    import keras.ops as ops
    result = ops.relu(tensor)
"""

__version__ = "1.0.0"

from . import distribution
from . import ops
from . import random


def _get_keras_version():
    """Get the major version of Keras."""
    try:
        import keras
        version = keras.__version__
        major_version = int(version.split('.')[0])
        return major_version
    except Exception:
        return None


def _is_keras_3():
    """Check if the installed Keras is version 3 or higher."""
    version = _get_keras_version()
    return version is not None and version >= 3


def apply_patch():
    """
    Apply the keras.ops compatibility patch.
    
    This function attaches the ops module to keras.ops if running Keras 2.
    When running Keras 3, this function does nothing since keras.ops is already available.
    
    This function should be called early in your code, before importing keras.ops.
    
    Example:
        from legacy_keras_patch import apply_patch
        apply_patch()
        
        import keras.ops as ops
        x = ops.ones((3, 3))
        y = ops.relu(x)
    """
    if _is_keras_3():
        # Keras 3 already has keras.ops, no need to patch
        return
    
    try:
        import keras
        
        # Check if keras.ops already exists
        if not hasattr(keras, 'ops'):
            # Attach our ops module to keras
            keras.ops = ops
        
        # Check if keras.random already exists
        if not hasattr(keras, 'random'):
            # Attach our random module to keras
            keras.random = random
        
        # Check if keras.distribution already exists
        if not hasattr(keras, 'distribution'):
            # Attach our distribution module to keras
            keras.distribution = distribution
        
    except ImportError:
        raise ImportError(
            "Keras is not installed. Please install Keras 2 or Keras 3 "
            "before using LegacyKerasPatch."
        )


def is_patched():
    """
    Check if the keras.ops patch has been applied.
    
    Returns:
        bool: True if keras.ops is available (either patched or native Keras 3).
    """
    try:
        import keras
        return hasattr(keras, 'ops')
    except ImportError:
        return False


def get_keras_version():
    """
    Get the major version of the installed Keras.
    
    Returns:
        int or None: The major version number (2 or 3), or None if Keras is not installed.
    """
    return _get_keras_version()


def is_keras_3():
    """
    Check if the installed Keras is version 3 or higher.
    
    Returns:
        bool: True if Keras 3 or higher is installed.
    """
    return _is_keras_3()


# Export public API
__all__ = [
    'apply_patch',
    'is_patched', 
    'get_keras_version',
    'is_keras_3',
    'distribution',
    'ops',
    'random',
]
