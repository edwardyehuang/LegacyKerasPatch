# LegacyKerasPatch

A compatibility patch that provides Keras 3's `keras.ops` and `keras.random` modules for Keras 2 + TensorFlow users.

## Overview

LegacyKerasPatch enables Keras 2 users to migrate their code to use `keras.ops` and `keras.random`, preparing for a future switch to Keras 3. The patch simulates Keras 3's operations by wrapping corresponding TensorFlow functions.

**Key Features:**
- Provides the complete `keras.ops` and `keras.random` APIs available in Keras 3
- Wraps TensorFlow operations (tf.nn, tf.linalg, tf.image, tf.random, etc.)
- Automatically detects Keras version - does nothing on Keras 3
- Manual activation via function call
- **Editor autocomplete support:** Includes type stubs so that editors like VS Code provide autocomplete and type hints for `keras.ops` and `keras.random` statically — even before running `apply_patch()`

## Installation

```bash
pip install legacy-keras-patch
```

Or install from GitHub directly:

```bash
pip install git+https://github.com/edwardyehuang/LegacyKerasPatch.git
```

Or install from source (editable/development mode):

```bash
git clone https://github.com/edwardyehuang/LegacyKerasPatch.git
cd LegacyKerasPatch
pip install -e .
```

## Quick Start

```python
# Apply the patch before importing keras.ops or keras.random
from legacy_keras_patch import apply_patch
apply_patch()

# Now you can use keras.ops as in Keras 3
import keras.ops as ops

# Basic operations
x = ops.ones((3, 3))
y = ops.relu(x)
z = ops.matmul(x, y)

# Neural network operations
logits = ops.nn.softmax(z)

# Image operations
import keras.ops.image as image_ops
resized = image_ops.resize(images, (224, 224))

# Linear algebra
eigenvalues, eigenvectors = ops.linalg.eigh(matrix)

# Now you can also use keras.random as in Keras 3
import keras.random as random

# Random sampling
samples = random.normal((3, 3), mean=0.0, stddev=1.0)
uniform_vals = random.uniform((3, 3), minval=0.0, maxval=1.0)

# Seed management
seed_gen = random.SeedGenerator(42)
x = random.normal((3, 3), seed=seed_gen)

# Dropout
dropped = random.dropout(tensor, rate=0.5)
```

## API Reference

### Main Module Functions

#### `apply_patch()`
Applies the keras.ops and keras.random compatibility patch. Call this before using keras.ops or keras.random.

```python
from legacy_keras_patch import apply_patch
apply_patch()
```

#### `is_patched()`
Returns `True` if keras.ops is available (either patched or native Keras 3).

#### `get_keras_version()`
Returns the major version of installed Keras (2 or 3), or `None` if not installed.

#### `is_keras_3()`
Returns `True` if Keras 3 or higher is installed.

### Available Operations

The patch provides all operations available in `keras.ops` in Keras 3:

#### Core Operations (`ops.*`)
- **Tensor creation:** `zeros`, `ones`, `full`, `empty`, `eye`, `arange`, `linspace`, etc.
- **Math operations:** `abs`, `sqrt`, `exp`, `log`, `sin`, `cos`, `matmul`, `dot`, etc.
- **Reductions:** `sum`, `mean`, `max`, `min`, `prod`, `all`, `any`, etc.
- **Array manipulation:** `reshape`, `transpose`, `concatenate`, `stack`, `split`, etc.
- **Activations:** `relu`, `sigmoid`, `softmax`, `gelu`, `silu`, `tanh`, etc.
- **FFT:** `fft`, `fft2`, `rfft`, `irfft`, `stft`, `istft`
- **Control flow:** `cond`, `while_loop`, `fori_loop`, `scan`, `map`

#### Neural Network (`ops.nn.*`)
- **Pooling:** `max_pool`, `average_pool`, `adaptive_average_pool`, `adaptive_max_pool`
- **Convolutions:** `conv`, `conv_transpose`, `depthwise_conv`, `separable_conv`
- **Activations:** `relu`, `relu6`, `sigmoid`, `softmax`, `gelu`, `silu`, `elu`, `selu`, etc.
- **Normalization:** `batch_normalization`, `layer_normalization`, `rms_normalization`
- **Loss functions:** `binary_crossentropy`, `categorical_crossentropy`, `sparse_categorical_crossentropy`, `ctc_loss`
- **Attention:** `dot_product_attention`
- **Encoding:** `one_hot`, `multi_hot`

#### Image Operations (`ops.image.*`)
- `resize` - Resize images
- `crop_images` - Crop images
- `pad_images` - Pad images
- `affine_transform` - Apply affine transformations
- `extract_patches` - Extract image patches
- `gaussian_blur` - Apply Gaussian blur
- `rgb_to_grayscale`, `rgb_to_hsv`, `hsv_to_rgb` - Color space conversions

#### Linear Algebra (`ops.linalg.*`)
- `cholesky`, `cholesky_inverse` - Cholesky decomposition
- `det`, `logdet` - Determinant
- `eig`, `eigh` - Eigendecomposition
- `inv` - Matrix inverse
- `svd` - Singular value decomposition
- `qr` - QR decomposition
- `solve`, `solve_triangular`, `lstsq` - Linear system solvers
- `norm` - Matrix/vector norms

#### NumPy-like Operations (`ops.numpy.*`)
Full NumPy-compatible API for tensor operations.

#### Random Operations (`keras.random.*`)
- **`SeedGenerator`** - Seed management class for reproducible random operations
- **Distributions:** `normal`, `uniform`, `truncated_normal`, `gamma`, `beta`, `binomial`
- **Discrete sampling:** `categorical`, `randint`
- **Utilities:** `shuffle`, `dropout`

## Compatibility

- **Keras 2.x + TensorFlow 2.x:** Full support via the patch
- **Keras 3.x:** No action needed - keras.ops and keras.random are native

## Behavior

When you call `apply_patch()`:

1. **If Keras 3 is detected:** The function returns immediately without making any changes, since `keras.ops` already exists natively.

2. **If Keras 2 is detected:** The function attaches the compatibility `ops` module to `keras.ops` and the `random` module to `keras.random`, making all operations available.

## Example Migration

Before (Keras 2 with TensorFlow):
```python
import tensorflow as tf

x = tf.nn.relu(tensor)
y = tf.linalg.matmul(a, b)
z = tf.image.resize(images, [224, 224])
samples = tf.random.normal((3, 3))
```

After (with LegacyKerasPatch or Keras 3):
```python
from legacy_keras_patch import apply_patch
apply_patch()

import keras.ops as ops
import keras.random as random

x = ops.relu(tensor)
y = ops.matmul(a, b)
z = ops.image.resize(images, (224, 224))
samples = random.normal((3, 3))
```

The code after migration will work with both Keras 2 (with the patch) and Keras 3 (natively).

## License

MIT License