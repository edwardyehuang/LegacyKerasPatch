"""
Linear algebra operations for Keras 2 compatibility.

This module provides keras.ops.linalg compatible operations by wrapping TensorFlow functions.
"""

import tensorflow as tf


def cholesky(x):
    """Compute the Cholesky decomposition of a positive definite matrix."""
    return tf.linalg.cholesky(x)


def cholesky_inverse(x):
    """Compute the inverse of a matrix using its Cholesky decomposition."""
    # If x is already the Cholesky factor L, compute inverse of L @ L.T
    # inv(A) = inv(L.T) @ inv(L)
    L_inv = tf.linalg.inv(x)
    return tf.matmul(L_inv, L_inv, transpose_a=True)


def det(x):
    """Compute the determinant of a matrix."""
    return tf.linalg.det(x)


def eig(x):
    """Compute eigenvalues and eigenvectors of a matrix."""
    return tf.linalg.eig(x)


def eigh(x):
    """Compute eigenvalues and eigenvectors of a Hermitian matrix."""
    return tf.linalg.eigh(x)


def inv(x):
    """Compute the inverse of a matrix."""
    return tf.linalg.inv(x)


def jvp(primals, tangents, fn):
    """Compute Jacobian-vector products."""
    # This is a simplified implementation using forward-mode autodiff
    with tf.GradientTape() as tape:
        tape.watch(primals)
        result = fn(*primals)
    
    # Compute JVP using the chain rule
    jvps = tape.jacobian(result, primals)
    
    output_tangents = []
    for jvp_matrix, tangent in zip(jvps, tangents):
        if jvp_matrix is not None and tangent is not None:
            output_tangents.append(tf.tensordot(jvp_matrix, tangent, axes=len(tangent.shape)))
        else:
            output_tangents.append(None)
    
    return result, output_tangents


def lstsq(a, b, rcond=None):
    """Solve least squares problems."""
    return tf.linalg.lstsq(a, b, l2_regularizer=rcond if rcond is not None else 0.0)


def lu_factor(x):
    """Compute the LU factorization of a matrix."""
    return tf.linalg.lu(x)


def norm(x, ord=None, axis=None, keepdims=False):
    """Compute the matrix or vector norm."""
    return tf.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)


def qr(x, mode="reduced"):
    """Compute the QR decomposition of a matrix."""
    full_matrices = mode == "complete"
    return tf.linalg.qr(x, full_matrices=full_matrices)


def solve(a, b):
    """Solve a linear system of equations."""
    return tf.linalg.solve(a, b)


def solve_triangular(a, b, lower=True):
    """Solve a triangular linear system."""
    return tf.linalg.triangular_solve(a, b, lower=lower)


def svd(x, full_matrices=True, compute_uv=True):
    """Compute the singular value decomposition of a matrix."""
    if compute_uv:
        s, u, v = tf.linalg.svd(x, full_matrices=full_matrices, compute_uv=True)
        return u, s, v
    else:
        return tf.linalg.svd(x, full_matrices=full_matrices, compute_uv=False)
