"""
Generic Extended Kalman Filter (EKF) and Unscented Kalman Filter (UKF)
Pure algorithm implementations - no system-specific code

Fixed: Robust Cholesky decomposition with regularization
"""

import numpy as np
from scipy.linalg import cholesky


class ExtendedKalmanFilter:
    """
    Generic Extended Kalman Filter implementation
    User must provide state transition and measurement functions
    """
    
    def __init__(self, state_dim):
        """
        Initialize EKF
        
        Args:
            state_dim: Dimension of state vector
        """
        self.n = state_dim
        self.x = None  # State estimate
        self.P = None  # State covariance
        
    def initialize(self, initial_state, initial_covariance):
        """
        Initialize state and covariance
        
        Args:
            initial_state: Initial state vector (n,)
            initial_covariance: Initial covariance matrix (n, n)
        """
        self.x = np.array(initial_state, dtype=float).reshape(-1)
        self.P = np.array(initial_covariance, dtype=float)
        
        # Ensure covariance is symmetric and positive definite
        self.P = 0.5 * (self.P + self.P.T)  # Force symmetry
        
    def predict(self, F, Q):
        """
        Prediction step
        
        Args:
            F: State transition matrix (n, n)
            Q: Process noise covariance (n, n)
        """
        # Predict state
        self.x = F @ self.x
        
        # Predict covariance
        self.P = F @ self.P @ F.T + Q
        
        # Ensure symmetry and positive definiteness
        self.P = 0.5 * (self.P + self.P.T)
        
    def update(self, measurement, H, R):
        """
        Update step with measurement
        
        Args:
            measurement: Measurement vector (m,)
            H: Jacobian of measurement function (m, n)
            R: Measurement covariance (m, m)
        """
        # Innovation covariance
        S = H @ self.P @ H.T + R
        
        # Ensure S is symmetric
        S = 0.5 * (S + S.T)
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ measurement
        
        # Update covariance (Joseph form for numerical stability)
        I = np.eye(self.n)
        I_KH = I - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        
        # Ensure symmetry and positive definiteness
        self.P = 0.5 * (self.P + self.P.T)


class UnscentedKalmanFilter:
    """
    Generic Unscented Kalman Filter implementation
    Uses sigma points to handle nonlinearity
    """
    
    def __init__(self, state_dim, alpha=0.001, beta=2.0, kappa=0.0):
        """
        Initialize UKF
        
        Args:
            state_dim: Dimension of state vector
            alpha: Sigma point spread parameter (1e-4 to 1)
            beta: Prior knowledge parameter (2 for Gaussian)
            kappa: Secondary scaling parameter (typically 0 or 3-n)
        """
        self.n = state_dim
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        
        # Calculate lambda
        self.lambda_ = alpha**2 * (self.n + kappa) - self.n
        
        # Calculate weights
        self.weights_m, self.weights_c = self._calculate_weights()
        
        # State
        self.x = None
        self.P = None
        self.sigma_points_pred = None
        
    def initialize(self, initial_state, initial_covariance):
        """
        Initialize state and covariance
        
        Args:
            initial_state: Initial state vector (n,)
            initial_covariance: Initial covariance matrix (n, n)
        """
        self.x = np.array(initial_state, dtype=float).reshape(-1)
        self.P = np.array(initial_covariance, dtype=float)
        
        # Ensure covariance is symmetric and positive definite
        self.P = 0.5 * (self.P + self.P.T)
        
    def _calculate_weights(self):
        """Calculate sigma point weights for mean and covariance"""
        n = self.n
        lambda_ = self.lambda_
        
        # Weights for mean
        weights_m = np.zeros(2 * n + 1)
        weights_m[0] = lambda_ / (n + lambda_)
        weights_m[1:] = 1.0 / (2.0 * (n + lambda_))
        
        # Weights for covariance
        weights_c = np.zeros(2 * n + 1)
        weights_c[0] = lambda_ / (n + lambda_) + (1 - self.alpha**2 + self.beta)
        weights_c[1:] = 1.0 / (2.0 * (n + lambda_))
        
        return weights_m, weights_c
        
    def generate_sigma_points(self, mean, covariance):
        """
        Generate sigma points using Cholesky decomposition
        
        FIXED: Robust implementation with multiple fallback strategies
        
        Args:
            mean: Mean vector (n,)
            covariance: Covariance matrix (n, n)
            
        Returns:
            Sigma points matrix (2n+1, n)
        """
        n = self.n
        lambda_ = self.lambda_
        
        sigma_points = np.zeros((2 * n + 1, n))
        sigma_points[0] = mean
        
        # Matrix to decompose
        A = (n + lambda_) * covariance
        
        # Ensure symmetry
        A = 0.5 * (A + A.T)
        
        # Try Cholesky decomposition with multiple strategies
        L = None
        
        # Strategy 1: Direct Cholesky
        try:
            L = cholesky(A, lower=True)
        except np.linalg.LinAlgError:
            pass
        
        # Strategy 2: Add small regularization
        if L is None:
            try:
                epsilon = 1e-9
                A_reg = A + epsilon * np.eye(n)
                L = cholesky(A_reg, lower=True)
            except np.linalg.LinAlgError:
                pass
        
        # Strategy 3: Larger regularization
        if L is None:
            try:
                epsilon = 1e-6
                A_reg = A + epsilon * np.eye(n)
                L = cholesky(A_reg, lower=True)
            except np.linalg.LinAlgError:
                pass
        
        # Strategy 4: Eigenvalue decomposition (always works)
        if L is None:
            # Use eigenvalue decomposition as fallback
            eigenvalues, eigenvectors = np.linalg.eigh(A)
            
            # Ensure all eigenvalues are positive
            eigenvalues = np.maximum(eigenvalues, 1e-9)
            
            # Compute L = Q * sqrt(Lambda)
            L = eigenvectors @ np.diag(np.sqrt(eigenvalues))
        
        # Generate sigma points
        for i in range(n):
            sigma_points[i + 1] = mean + L[:, i]
            sigma_points[n + i + 1] = mean - L[:, i]
            
        return sigma_points
        
    def predict(self, state_transition_fn, Q):
        """
        Prediction step using Unscented Transform
        
        Args:
            state_transition_fn: Function that takes state and returns predicted state
                                 Signature: f(state) -> predicted_state
            Q: Process noise covariance (n, n)
        """
        # Generate sigma points from current state
        sigma_points = self.generate_sigma_points(self.x, self.P)
        
        # Propagate sigma points through state transition function
        sigma_points_pred = np.zeros_like(sigma_points)
        for i in range(sigma_points.shape[0]):
            sigma_points_pred[i] = state_transition_fn(sigma_points[i])
        
        # Compute predicted mean
        self.x = np.sum(self.weights_m[:, np.newaxis] * sigma_points_pred, axis=0)
        
        # Compute predicted covariance
        self.P = Q.copy()
        for i in range(sigma_points_pred.shape[0]):
            diff = sigma_points_pred[i] - self.x
            self.P += self.weights_c[i] * np.outer(diff, diff)
        
        # Ensure symmetry and positive definiteness
        self.P = 0.5 * (self.P + self.P.T)
        
        # Store predicted sigma points for update step
        self.sigma_points_pred = sigma_points_pred
        
    def update(self, measurement, measurement_fn, R, angle_indices=None):
        """
        Update step with measurement using Unscented Transform
        
        Args:
            measurement: Measurement vector (m,)
            measurement_fn: Function that transforms state to measurement space
                           Signature: h(state) -> measurement
            R: Measurement noise covariance (m, m)
            angle_indices: List of measurement indices that are angles (for normalization)
        """
        if angle_indices is None:
            angle_indices = []
            
        n_sigma = self.sigma_points_pred.shape[0]
        m = len(measurement)
        
        # Transform sigma points to measurement space
        Z = np.zeros((n_sigma, m))
        for i in range(n_sigma):
            Z[i] = measurement_fn(self.sigma_points_pred[i])
        
        # Predicted measurement mean
        z_pred = np.sum(self.weights_m[:, np.newaxis] * Z, axis=0)
        
        # Innovation covariance Pzz
        Pzz = R.copy()
        for i in range(n_sigma):
            diff = Z[i] - z_pred
            # Normalize angles if specified
            for idx in angle_indices:
                diff[idx] = self._normalize_angle(diff[idx])
            Pzz += self.weights_c[i] * np.outer(diff, diff)
        
        # Ensure Pzz is symmetric
        Pzz = 0.5 * (Pzz + Pzz.T)
        
        # Cross-covariance Pxz
        Pxz = np.zeros((self.n, m))
        for i in range(n_sigma):
            x_diff = self.sigma_points_pred[i] - self.x
            z_diff = Z[i] - z_pred
            # Normalize angles if specified
            for idx in angle_indices:
                z_diff[idx] = self._normalize_angle(z_diff[idx])
            Pxz += self.weights_c[i] * np.outer(x_diff, z_diff)
        
        # Kalman gain
        K = Pxz @ np.linalg.inv(Pzz)
        
        # Innovation
        y = measurement - z_pred
        # Normalize angle innovations
        for idx in angle_indices:
            y[idx] = self._normalize_angle(y[idx])
        
        # Update state
        self.x = self.x + K @ y
        
        # Update covariance
        self.P = self.P - K @ Pzz @ K.T
        
        # Ensure symmetry and positive definiteness
        self.P = 0.5 * (self.P + self.P.T)
        
    @staticmethod
    def _normalize_angle(angle):
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle