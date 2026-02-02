"""Tests for linear algebra operations in legacy_keras_patch.ops.linalg."""

import pytest
import numpy as np
import tensorflow as tf

from legacy_keras_patch.ops import linalg


class TestDecompositions:
    """Test matrix decomposition operations."""
    
    def test_cholesky(self):
        """Test Cholesky decomposition."""
        # Create a positive definite matrix
        a = tf.constant([[4.0, 2.0], [2.0, 3.0]])
        matrix = tf.matmul(a, tf.transpose(a)) + tf.eye(2)
        result = linalg.cholesky(matrix)
        # Verify L @ L.T = matrix
        reconstructed = tf.matmul(result, tf.transpose(result))
        np.testing.assert_array_almost_equal(reconstructed.numpy(), matrix.numpy(), decimal=5)
    
    def test_qr(self):
        """Test QR decomposition."""
        x = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        q, r = linalg.qr(x)
        # Verify Q @ R = X
        reconstructed = tf.matmul(q, r)
        np.testing.assert_array_almost_equal(reconstructed.numpy(), x.numpy(), decimal=5)
    
    def test_qr_full(self):
        """Test QR decomposition with full matrices."""
        x = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        q, r = linalg.qr(x, mode="complete")
        # Q should be (3, 3) for complete mode
        assert q.shape == (3, 3)
    
    def test_svd(self):
        """Test SVD decomposition."""
        x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        u, s, v = linalg.svd(x)
        # Verify U @ S @ V.T ≈ X
        s_matrix = tf.linalg.diag(s)
        reconstructed = tf.matmul(tf.matmul(u, s_matrix), tf.transpose(v))
        np.testing.assert_array_almost_equal(reconstructed.numpy(), x.numpy(), decimal=5)
    
    def test_eigh(self):
        """Test eigendecomposition of Hermitian matrix."""
        # Symmetric matrix
        x = tf.constant([[4.0, 2.0], [2.0, 3.0]])
        eigenvalues, eigenvectors = linalg.eigh(x)
        # Verify A @ v = λ * v for each eigenpair
        for i in range(2):
            lhs = tf.matmul(x, eigenvectors[:, i:i+1])
            rhs = eigenvalues[i] * eigenvectors[:, i:i+1]
            np.testing.assert_array_almost_equal(lhs.numpy(), rhs.numpy(), decimal=5)
    
    def test_eig(self):
        """Test eigendecomposition of general matrix."""
        x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        eigenvalues, eigenvectors = linalg.eig(x)
        # Just verify it runs and returns correct shapes
        assert eigenvalues.shape == (2,)
        assert eigenvectors.shape == (2, 2)
    
    def test_lu_factor(self):
        """Test LU factorization."""
        x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        lu, p = linalg.lu_factor(x)
        # Just verify it runs and returns correct shapes
        assert lu.shape == x.shape
        assert p.shape == (2,)


class TestDeterminant:
    """Test determinant operations."""
    
    def test_det(self):
        """Test determinant."""
        x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        result = linalg.det(x)
        expected = 1.0 * 4.0 - 2.0 * 3.0  # -2.0
        np.testing.assert_almost_equal(result.numpy(), expected)
    
    def test_det_identity(self):
        """Test determinant of identity matrix."""
        x = tf.eye(3)
        result = linalg.det(x)
        np.testing.assert_almost_equal(result.numpy(), 1.0)


class TestInverse:
    """Test matrix inverse operations."""
    
    def test_inv(self):
        """Test matrix inverse."""
        x = tf.constant([[4.0, 7.0], [2.0, 6.0]])
        inv = linalg.inv(x)
        # Verify A @ A_inv = I
        identity = tf.matmul(x, inv)
        np.testing.assert_array_almost_equal(identity.numpy(), np.eye(2), decimal=5)
    
    def test_cholesky_inverse(self):
        """Test Cholesky inverse."""
        # Create a positive definite matrix and its Cholesky factor
        a = tf.constant([[4.0, 2.0], [2.0, 3.0]])
        matrix = tf.matmul(a, tf.transpose(a)) + tf.eye(2)
        L = linalg.cholesky(matrix)
        
        # Compute inverse using Cholesky
        inv = linalg.cholesky_inverse(L)
        
        # Verify matrix @ inv ≈ I
        identity = tf.matmul(matrix, inv)
        np.testing.assert_array_almost_equal(identity.numpy(), np.eye(2), decimal=5)


class TestLinearSolvers:
    """Test linear equation solvers."""
    
    def test_solve(self):
        """Test solving linear system Ax = b."""
        A = tf.constant([[3.0, 1.0], [1.0, 2.0]])
        b = tf.constant([[9.0], [8.0]])
        x = linalg.solve(A, b)
        # Verify A @ x = b
        result = tf.matmul(A, x)
        np.testing.assert_array_almost_equal(result.numpy(), b.numpy(), decimal=5)
    
    def test_solve_multiple_rhs(self):
        """Test solving with multiple right-hand sides."""
        A = tf.constant([[3.0, 1.0], [1.0, 2.0]])
        b = tf.constant([[9.0, 5.0], [8.0, 3.0]])
        x = linalg.solve(A, b)
        # Verify A @ x = b
        result = tf.matmul(A, x)
        np.testing.assert_array_almost_equal(result.numpy(), b.numpy(), decimal=5)
    
    def test_solve_triangular_lower(self):
        """Test solving triangular system (lower)."""
        A = tf.constant([[1.0, 0.0], [2.0, 1.0]])
        b = tf.constant([[1.0], [4.0]])
        x = linalg.solve_triangular(A, b, lower=True)
        # Verify A @ x = b
        result = tf.matmul(A, x)
        np.testing.assert_array_almost_equal(result.numpy(), b.numpy(), decimal=5)
    
    def test_solve_triangular_upper(self):
        """Test solving triangular system (upper)."""
        A = tf.constant([[1.0, 2.0], [0.0, 1.0]])
        b = tf.constant([[5.0], [3.0]])
        x = linalg.solve_triangular(A, b, lower=False)
        # Verify A @ x = b
        result = tf.matmul(A, x)
        np.testing.assert_array_almost_equal(result.numpy(), b.numpy(), decimal=5)
    
    def test_lstsq(self):
        """Test least squares solver."""
        A = tf.constant([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
        b = tf.constant([[1.0], [2.0], [2.0]])
        x = linalg.lstsq(A, b)
        # Just verify it runs and returns correct shape
        assert x.shape == (2, 1)


class TestNorm:
    """Test norm operations."""
    
    def test_norm_vector_l2(self):
        """Test L2 norm of a vector."""
        x = tf.constant([3.0, 4.0])
        result = linalg.norm(x)
        np.testing.assert_almost_equal(result.numpy(), 5.0)
    
    def test_norm_vector_l1(self):
        """Test L1 norm of a vector."""
        x = tf.constant([3.0, 4.0])
        result = linalg.norm(x, ord=1)
        np.testing.assert_almost_equal(result.numpy(), 7.0)
    
    def test_norm_vector_inf(self):
        """Test infinity norm of a vector."""
        x = tf.constant([3.0, -4.0])
        result = linalg.norm(x, ord=np.inf)
        np.testing.assert_almost_equal(result.numpy(), 4.0)
    
    def test_norm_matrix_fro(self):
        """Test Frobenius norm of a matrix."""
        x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        result = linalg.norm(x, ord='fro')
        expected = np.sqrt(1 + 4 + 9 + 16)
        np.testing.assert_almost_equal(result.numpy(), expected, decimal=5)
    
    def test_norm_axis(self):
        """Test norm along axis."""
        x = tf.constant([[3.0, 4.0], [5.0, 12.0]])
        result = linalg.norm(x, axis=1)
        expected = np.array([5.0, 13.0])
        np.testing.assert_array_almost_equal(result.numpy(), expected)
    
    def test_norm_keepdims(self):
        """Test norm with keepdims."""
        x = tf.constant([[3.0, 4.0], [5.0, 12.0]])
        result = linalg.norm(x, axis=1, keepdims=True)
        assert result.shape == (2, 1)
