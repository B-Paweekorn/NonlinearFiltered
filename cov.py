import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

np.random.seed(42)

# Simulation parameters
n_samples = 1000
r_true = 1.0
theta_true = 1.0

# Radar measurement noise in polar coordinates
sigma_r = 0.1
sigma_theta = 0.5

# Generate samples in polar coordinates
r_samples = r_true + np.random.randn(n_samples) * sigma_r
theta_samples = theta_true + np.random.randn(n_samples) * sigma_theta

# Convert to Cartesian
x_samples = r_samples * np.cos(theta_samples)
y_samples = r_samples * np.sin(theta_samples)

# True position in Cartesian
true_x = r_true * np.cos(theta_true)
true_y = r_true * np.sin(theta_true)

# Compute sample statistics
samples_cartesian = np.column_stack([x_samples, y_samples])
true_mean = np.mean(samples_cartesian, axis=0)
true_cov = np.cov(samples_cartesian.T)

# EKF: Jacobian-based linearization
H = np.array([
    [np.cos(theta_true), -r_true * np.sin(theta_true)],
    [np.sin(theta_true),  r_true * np.cos(theta_true)]
])

R_polar = np.diag([sigma_r**2, sigma_theta**2])
ekf_cov = H @ R_polar @ H.T
ekf_mean = np.array([true_x, true_y])

# UKF: Unscented transform
def compute_ukf_covariance(mean_r, mean_theta, sigma_r, sigma_theta, 
                           alpha=0.001, beta=2.0, kappa=0.0):
    n = 2
    lambda_ = alpha**2 * (n + kappa) - n
    
    # Compute weights
    Wm = np.zeros(2*n + 1)
    Wm[0] = lambda_ / (n + lambda_)
    Wm[1:] = 1.0 / (2.0 * (n + lambda_))
    
    Wc = np.zeros(2*n + 1)
    Wc[0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)
    Wc[1:] = 1.0 / (2.0 * (n + lambda_))
     
    mean_polar = np.array([mean_r, mean_theta])
    P_polar = np.diag([sigma_r**2, sigma_theta**2])
    
    # Generate sigma points
    sigma_points_polar = np.zeros((2*n + 1, n))
    sigma_points_polar[0] = mean_polar
    
    try:
        L = np.linalg.cholesky((n + lambda_) * P_polar)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh((n + lambda_) * P_polar)
        L = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 0)))
    
    for i in range(n):
        sigma_points_polar[i + 1] = mean_polar + L[:, i]
        sigma_points_polar[i + 1 + n] = mean_polar - L[:, i]
    
    # Transform to Cartesian
    sigma_points_cart = np.zeros((2*n + 1, 2))
    for i in range(2*n + 1):
        r = sigma_points_polar[i, 0]
        theta = sigma_points_polar[i, 1]
        sigma_points_cart[i, 0] = r * np.cos(theta)
        sigma_points_cart[i, 1] = r * np.sin(theta)
    
    ukf_mean = np.sum(Wm[:, np.newaxis] * sigma_points_cart, axis=0)
    
    ukf_cov = np.zeros((2, 2))
    for i in range(2*n + 1):
        diff = sigma_points_cart[i] - ukf_mean
        ukf_cov += Wc[i] * np.outer(diff, diff)
    
    return ukf_cov, ukf_mean, sigma_points_polar, sigma_points_cart

ukf_cov, ukf_mean, sigma_points_polar, sigma_points_cart = compute_ukf_covariance(
    r_true, theta_true, sigma_r, sigma_theta
)

def plot_covariance_ellipse(ax, mean, cov, color, label, n_std=2, 
                            linestyle='-', linewidth=2.5):
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * n_std * np.sqrt(eigenvalues)
    ellipse = Ellipse(mean, width, height, angle=angle, 
                     facecolor='none', edgecolor=color, linewidth=linewidth, 
                     label=label, linestyle=linestyle)
    ax.add_patch(ellipse)

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Polar space
ax1.scatter(r_samples, theta_samples, alpha=0.3, s=10, c='blue', 
           label='Random samples')
plot_covariance_ellipse(ax1, [r_true, theta_true], R_polar, 'green', 
                       'Covariance', n_std=2, linestyle='-', linewidth=1.5)
ax1.scatter(sigma_points_polar[:, 0], sigma_points_polar[:, 1], 
           c='red', s=120, marker='x', linewidths=2.5, 
           label='UKF sigma points', zorder=4)
ax1.set_xlabel('r (m)', fontsize=12, fontweight='bold')
ax1.set_ylabel('θ (rad)', fontsize=12, fontweight='bold')
ax1.set_title('Polar Coordinates', fontsize=14, fontweight='bold', color='darkred')
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)

# Cartesian space
ax2.scatter(x_samples, y_samples, alpha=0.3, s=10, c='blue', 
           label='Random samples')
plot_covariance_ellipse(ax2, true_mean, true_cov, 'darkblue', 
                       'True covariance', n_std=2, linestyle='-', linewidth=3)
plot_covariance_ellipse(ax2, ekf_mean, ekf_cov, 'green', 
                       'EKF linearized', n_std=2, linestyle='-', linewidth=1.5)
plot_covariance_ellipse(ax2, ukf_mean, ukf_cov, 'red', 
                       'UKF (sigma points)', n_std=2, linestyle='-', linewidth=1.5)
ax2.scatter([true_x], [true_y], c='black', s=400, marker='*', 
           edgecolors='white', linewidths=1, label='True mean', zorder=4)
ax2.scatter(sigma_points_cart[:, 0], sigma_points_cart[:, 1], 
           c='red', s=120, marker='x', linewidths=2, 
           label='UKF sigma points', zorder=5)
ax2.set_xlabel('x (m)', fontsize=12, fontweight='bold')
ax2.set_ylabel('y (m)', fontsize=12, fontweight='bold')
ax2.set_title('Cartesian Coordinates', fontsize=14, fontweight='bold', color='darkred')
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')

plt.tight_layout()
plt.savefig('ekf_ukf_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# Numerical results
print("\n" + "="*60)
print("COVARIANCE COMPARISON")
print("="*60)

print("\nTrue Covariance (Monte Carlo):")
print(true_cov)
print(f"Trace: {np.trace(true_cov):.6f}, Det: {np.linalg.det(true_cov):.6f}")

print("\nEKF Covariance (Jacobian):")
print(ekf_cov)
print(f"Trace: {np.trace(ekf_cov):.6f}, Det: {np.linalg.det(ekf_cov):.6f}")

print("\nUKF Covariance (Unscented Transform):")
print(ukf_cov)
print(f"Trace: {np.trace(ukf_cov):.6f}, Det: {np.linalg.det(ukf_cov):.6f}")

# Error metrics
ekf_error = np.linalg.norm(true_cov - ekf_cov, 'fro')
ukf_error = np.linalg.norm(true_cov - ukf_cov, 'fro')
improvement = (ekf_error - ukf_error) / ekf_error * 100

print("\n" + "="*60)
print("ERROR METRICS")
print("="*60)
print(f"EKF Frobenius error: {ekf_error:.6f}")
print(f"UKF Frobenius error: {ukf_error:.6f}")
print(f"Improvement: {improvement:.1f}%")
print("="*60)