"""
Extended Kalman Filter (EKF) and Unscented Kalman Filter (UKF)
Generic implementations for nonlinear state estimation
"""

import numpy as np
from scipy.linalg import cholesky


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for nonlinear systems.
    Uses Jacobian linearization for prediction and update steps.
    """
    
    def __init__(self, state_dim):
        self.n = state_dim
        self.x = None
        self.P = None
        
    def initialize(self, initial_state, initial_covariance):
        self.x = np.array(initial_state, dtype=float).reshape(-1)
        self.P = np.array(initial_covariance, dtype=float)
        self.P = 0.5 * (self.P + self.P.T)
        
    def predict(self, F, Q):
        """
        Prediction step.
        
        Args:
            F: State transition matrix (n x n)
            Q: Process noise covariance (n x n)
        """
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)
        
    def update(self, measurement, H, R):
        """
        Measurement update step.
        
        Args:
            measurement: Measurement vector (m,)
            H: Measurement Jacobian (m x n)
            R: Measurement noise covariance (m x m)
        """
        S = H @ self.P @ H.T + R
        S = 0.5 * (S + S.T)
        
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ measurement
        
        # Joseph form for numerical stability
        I = np.eye(self.n)
        I_KH = I - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)


class UnscentedKalmanFilter:
    """
    Unscented Kalman Filter for nonlinear systems.
    Uses sigma points to propagate mean and covariance through nonlinear functions.
    """
    
    def __init__(self, state_dim, alpha=0.001, beta=2.0, kappa=0.0):
        """
        Args:
            state_dim: State vector dimension
            alpha: Spread of sigma points (1e-4 to 1)
            beta: Prior distribution parameter (2 for Gaussian)
            kappa: Secondary scaling parameter (0 or 3-n)
        """
        self.n = state_dim
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        
        self.lambda_ = alpha**2 * (self.n + kappa) - self.n
        self.weights_m, self.weights_c = self._calculate_weights()
        
        self.x = None
        self.P = None
        self.sigma_points_pred = None
        
    def initialize(self, initial_state, initial_covariance):
        self.x = np.array(initial_state, dtype=float).reshape(-1)
        self.P = np.array(initial_covariance, dtype=float)
        self.P = 0.5 * (self.P + self.P.T)
        
    def _calculate_weights(self):
        """Calculate weights for mean and covariance reconstruction."""
        n = self.n
        lambda_ = self.lambda_
        
        weights_m = np.zeros(2 * n + 1)
        weights_m[0] = lambda_ / (n + lambda_)
        weights_m[1:] = 1.0 / (2.0 * (n + lambda_))
        
        weights_c = np.zeros(2 * n + 1)
        weights_c[0] = lambda_ / (n + lambda_) + (1 - self.alpha**2 + self.beta)
        weights_c[1:] = 1.0 / (2.0 * (n + lambda_))
        
        return weights_m, weights_c
        
    def generate_sigma_points(self, mean, covariance):
        """
        Generate sigma points around mean with specified covariance.
        Uses multiple fallback strategies for numerical robustness.
        
        Returns:
            Sigma points matrix (2n+1 x n)
        """
        n = self.n
        lambda_ = self.lambda_
        
        sigma_points = np.zeros((2 * n + 1, n))
        sigma_points[0] = mean
        
        A = (n + lambda_) * covariance
        A = 0.5 * (A + A.T)
        
        L = None
        
        # Try Cholesky decomposition
        try:
            L = cholesky(A, lower=True)
        except np.linalg.LinAlgError:
            pass
        
        # Fallback: add small regularization
        if L is None:
            try:
                A_reg = A + 1e-9 * np.eye(n)
                L = cholesky(A_reg, lower=True)
            except np.linalg.LinAlgError:
                pass
        
        # Final fallback: eigenvalue decomposition
        if L is None:
            eigenvalues, eigenvectors = np.linalg.eigh(A)
            eigenvalues = np.maximum(eigenvalues, 1e-9)
            L = eigenvectors @ np.diag(np.sqrt(eigenvalues))
        
        for i in range(n):
            sigma_points[i + 1] = mean + L[:, i]
            sigma_points[n + i + 1] = mean - L[:, i]
            
        return sigma_points
        
    def predict(self, state_transition_fn, Q):
        """
        Prediction step using unscented transform.
        
        Args:
            state_transition_fn: f(state) -> next_state
            Q: Process noise covariance (n x n)
        """
        sigma_points = self.generate_sigma_points(self.x, self.P)
        
        sigma_points_pred = np.zeros_like(sigma_points)
        for i in range(sigma_points.shape[0]):
            sigma_points_pred[i] = state_transition_fn(sigma_points[i])
        
        self.x = np.sum(self.weights_m[:, np.newaxis] * sigma_points_pred, axis=0)
        
        self.P = Q.copy()
        for i in range(sigma_points_pred.shape[0]):
            diff = sigma_points_pred[i] - self.x
            self.P += self.weights_c[i] * np.outer(diff, diff)
        
        self.P = 0.5 * (self.P + self.P.T)
        self.sigma_points_pred = sigma_points_pred
        
    def update(self, measurement, measurement_fn, R, angle_indices=None):
        """
        Measurement update using unscented transform.
        
        Args:
            measurement: Measurement vector (m,)
            measurement_fn: h(state) -> measurement
            R: Measurement noise covariance (m x m)
            angle_indices: Indices of angular measurements (for wrapping)
        """
        if angle_indices is None:
            angle_indices = []
            
        n_sigma = self.sigma_points_pred.shape[0]
        m = len(measurement)
        
        Z = np.zeros((n_sigma, m))
        for i in range(n_sigma):
            Z[i] = measurement_fn(self.sigma_points_pred[i])
        
        z_pred = np.sum(self.weights_m[:, np.newaxis] * Z, axis=0)
        
        Pzz = R.copy()
        for i in range(n_sigma):
            diff = Z[i] - z_pred
            for idx in angle_indices:
                diff[idx] = self._normalize_angle(diff[idx])
            Pzz += self.weights_c[i] * np.outer(diff, diff)
        
        Pzz = 0.5 * (Pzz + Pzz.T)
        
        Pxz = np.zeros((self.n, m))
        for i in range(n_sigma):
            x_diff = self.sigma_points_pred[i] - self.x
            z_diff = Z[i] - z_pred
            for idx in angle_indices:
                z_diff[idx] = self._normalize_angle(z_diff[idx])
            Pxz += self.weights_c[i] * np.outer(x_diff, z_diff)
        
        K = Pxz @ np.linalg.inv(Pzz)
        
        y = measurement - z_pred
        for idx in angle_indices:
            y[idx] = self._normalize_angle(y[idx])
        
        self.x = self.x + K @ y
        self.P = self.P - K @ Pzz @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        
    @staticmethod
    def _normalize_angle(angle):
        """Wrap angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle