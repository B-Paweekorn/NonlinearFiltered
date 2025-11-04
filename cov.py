import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 1000

# True state in POLAR coordinates
r_true = 1.0
theta_true = 1.0  # Small angle (about 57 degrees)

# Radar measurement noise (in polar coordinates)
sigma_r = 0.1
sigma_theta = 0.5  # Extremely large angular uncertainty to create banana shape

# Generate random samples in polar coordinates
r_samples = r_true + np.random.randn(n_samples) * sigma_r
theta_samples = theta_true + np.random.randn(n_samples) * sigma_theta

# Convert to Cartesian coordinates
x_samples = r_samples * np.cos(theta_samples)
y_samples = r_samples * np.sin(theta_samples)

# True position in Cartesian
true_x = r_true * np.cos(theta_true)
true_y = r_true * np.sin(theta_true)

# Compute true covariance from samples
samples_cartesian = np.column_stack([x_samples, y_samples])
true_mean = np.mean(samples_cartesian, axis=0)
true_cov = np.cov(samples_cartesian.T)

# =============================================================================
# EKF: Linearized Covariance using Jacobian
# =============================================================================
# Jacobian of transformation from polar to cartesian
# x = r * cos(theta), y = r * sin(theta)
# H = [cos(theta), -r*sin(theta)]
#     [sin(theta),  r*cos(theta)]
H = np.array([
    [np.cos(theta_true), -r_true * np.sin(theta_true)],
    [np.sin(theta_true),  r_true * np.cos(theta_true)]
])

# Measurement covariance in polar coordinates
R_polar = np.diag([sigma_r**2, sigma_theta**2])

# EKF linearized covariance in Cartesian coordinates
ekf_cov = H @ R_polar @ H.T
ekf_mean = np.array([true_x, true_y])

# =============================================================================
# UKF: Unscented Transform using Sigma Points
# =============================================================================
def compute_ukf_covariance(mean_r, mean_theta, sigma_r, sigma_theta, 
                           alpha=0.001, beta=2.0, kappa=0.0):
    """
    Compute UKF covariance using unscented transform
    
    Args:
        mean_r: Mean range
        mean_theta: Mean bearing
        sigma_r: Range standard deviation
        sigma_theta: Bearing standard deviation
        alpha, beta, kappa: UKF tuning parameters
    
    Returns:
        ukf_cov: Covariance in Cartesian coordinates
        ukf_mean: Mean in Cartesian coordinates
        sigma_points_polar: Sigma points in polar coordinates
        sigma_points_cart: Sigma points in Cartesian coordinates
    """
    # State dimension (2D: range and bearing)
    n = 2
    
    # UKF parameters
    lambda_ = alpha**2 * (n + kappa) - n
    
    # Weights for mean
    Wm = np.zeros(2*n + 1)
    Wm[0] = lambda_ / (n + lambda_)
    Wm[1:] = 1.0 / (2.0 * (n + lambda_))
    
    # Weights for covariance
    Wc = np.zeros(2*n + 1)
    Wc[0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)
    Wc[1:] = 1.0 / (2.0 * (n + lambda_))
     
    # Mean and covariance in polar coordinates
    mean_polar = np.array([mean_r, mean_theta])
    P_polar = np.diag([sigma_r**2, sigma_theta**2])
    
    # Generate sigma points in polar coordinates
    sigma_points_polar = np.zeros((2*n + 1, n))
    sigma_points_polar[0] = mean_polar
    
    # Compute matrix square root (Cholesky decomposition)
    try:
        L = np.linalg.cholesky((n + lambda_) * P_polar)
    except np.linalg.LinAlgError:
        # Fallback to eigenvalue decomposition if Cholesky fails
        eigvals, eigvecs = np.linalg.eigh((n + lambda_) * P_polar)
        L = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 0)))
    
    # Generate sigma points
    for i in range(n):
        sigma_points_polar[i + 1] = mean_polar + L[:, i]
        sigma_points_polar[i + 1 + n] = mean_polar - L[:, i]
    
    # Transform sigma points to Cartesian coordinates
    sigma_points_cart = np.zeros((2*n + 1, 2))
    for i in range(2*n + 1):
        r = sigma_points_polar[i, 0]
        theta = sigma_points_polar[i, 1]
        sigma_points_cart[i, 0] = r * np.cos(theta)
        sigma_points_cart[i, 1] = r * np.sin(theta)
    
    # Compute mean in Cartesian coordinates
    ukf_mean = np.sum(Wm[:, np.newaxis] * sigma_points_cart, axis=0)
    
    # Compute covariance in Cartesian coordinates
    ukf_cov = np.zeros((2, 2))
    for i in range(2*n + 1):
        diff = sigma_points_cart[i] - ukf_mean
        ukf_cov += Wc[i] * np.outer(diff, diff)
    
    return ukf_cov, ukf_mean, sigma_points_polar, sigma_points_cart

# Compute UKF covariance
ukf_cov, ukf_mean, sigma_points_polar, sigma_points_cart = compute_ukf_covariance(
    r_true, theta_true, sigma_r, sigma_theta
)

# Function to plot covariance ellipse
def plot_covariance_ellipse(ax, mean, cov, color, label, n_std=2, linestyle='-', linewidth=2.5):
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * n_std * np.sqrt(eigenvalues)
    ellipse = Ellipse(mean, width, height, angle=angle, 
                     facecolor='none', edgecolor=color, linewidth=linewidth, 
                     label=label, linestyle=linestyle)
    ax.add_patch(ellipse)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# =============================================================================
# Left plot: Polar coordinates
# =============================================================================
ax1.scatter(r_samples, theta_samples, 
           alpha=0.3, s=10, c='blue', label='random samples')

# Plot covariance ellipse in polar space
polar_cov = R_polar
plot_covariance_ellipse(ax1, [r_true, theta_true], polar_cov, 'green', 
                       'covariance', n_std=2, linestyle='-', linewidth=1.5)

ax1.scatter(sigma_points_polar[:, 0], sigma_points_polar[:, 1], 
           c='red', s=120, marker='x', linewidths=2.5, 
           label='UKF sigma points', zorder=4)



ax1.set_xlabel('r (m)', fontsize=12, fontweight='bold')
ax1.set_ylabel('θ (rad)', fontsize=12, fontweight='bold')
ax1.set_title('Random samples in polar coordinates', 
             fontsize=14, fontweight='bold', color='darkred')
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)

# =============================================================================
# Right plot: Cartesian coordinates
# =============================================================================
ax2.scatter(x_samples, y_samples, alpha=0.3, s=10, c='blue', 
           label='random samples')

# Plot true covariance ellipse (from Monte Carlo)
plot_covariance_ellipse(ax2, true_mean, true_cov, 'darkblue', 
                       'true covariance', n_std=2, linestyle='-', linewidth=3)

# Plot EKF linearized covariance ellipse (GREEN, SOLID LINE)
plot_covariance_ellipse(ax2, ekf_mean, ekf_cov, 'green', 
                       'EKF linearized', n_std=2, linestyle='-', linewidth=1.5)

# Plot UKF covariance ellipse (YELLOW, SOLID LINE)
plot_covariance_ellipse(ax2, ukf_mean, ukf_cov, 'red', 
                       'UKF (sigma points)', n_std=2, linestyle='-', linewidth=1.5)

# Plot true mean point in Cartesian space (BLACK STAR)
ax2.scatter([true_x], [true_y], 
           c='black', s=400, marker='*', 
           edgecolors='white', linewidths=1,
           label='true mean', zorder=4)

# Plot UKF sigma points in Cartesian space (RED X)
ax2.scatter(sigma_points_cart[:, 0], sigma_points_cart[:, 1], 
           c='red', s=120, marker='x', linewidths=2, 
           label='UKF sigma points', zorder=5)

ax2.set_xlabel('x (m)', fontsize=12, fontweight='bold')
ax2.set_ylabel('y (m)', fontsize=12, fontweight='bold')
ax2.set_title('Random samples in cartesian coordinates', 
             fontsize=14, fontweight='bold', color='darkred')
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')

plt.tight_layout()
plt.savefig('ekf_ukf_comparison_yellow_green.png', 
           dpi=150, bbox_inches='tight')
print("Plot saved successfully!")

plt.show()

# Print numerical comparison
print("\n" + "="*70)
print("COVARIANCE COMPARISON")
print("="*70)

print("\nTrue Mean (Cartesian):")
print(f"  x = {true_mean[0]:.6f}, y = {true_mean[1]:.6f}")

print("\nTrue Mean from polar transform:")
print(f"  x = {true_x:.6f}, y = {true_y:.6f}")

print("\nUKF Mean (weighted sigma points):")
print(f"  x = {ukf_mean[0]:.6f}, y = {ukf_mean[1]:.6f}")

print("\nEKF Mean (linearized):")
print(f"  x = {ekf_mean[0]:.6f}, y = {ekf_mean[1]:.6f}")

print("\n" + "-"*70)

print("\nTrue Covariance (Monte Carlo from samples):")
print(true_cov)
print(f"  Trace: {np.trace(true_cov):.6f}")
print(f"  Determinant: {np.linalg.det(true_cov):.6f}")

print("\nEKF Linearized Covariance (Jacobian method):")
print(ekf_cov)
print(f"  Trace: {np.trace(ekf_cov):.6f}")
print(f"  Determinant: {np.linalg.det(ekf_cov):.6f}")

print("\nUKF Covariance (Unscented Transform):")
print(ukf_cov)
print(f"  Trace: {np.trace(ukf_cov):.6f}")
print(f"  Determinant: {np.linalg.det(ukf_cov):.6f}")

print("\n" + "="*70)
print("ERROR ANALYSIS")
print("="*70)

# Mean errors
print("\nMean Errors (compared to true mean from polar):")
ekf_mean_error = np.linalg.norm(ekf_mean - np.array([true_x, true_y]))
ukf_mean_error = np.linalg.norm(ukf_mean - np.array([true_x, true_y]))
print(f"  EKF mean error: {ekf_mean_error:.6f} m")
print(f"  UKF mean error: {ukf_mean_error:.6f} m")

# EKF errors
ekf_error = true_cov - ekf_cov
ekf_frobenius = np.linalg.norm(ekf_error, 'fro')
ekf_trace_error = abs(np.trace(true_cov) - np.trace(ekf_cov))

print("\nEKF Error (True - EKF):")
print(ekf_error)
print(f"  Frobenius norm: {ekf_frobenius:.6f}")
print(f"  Trace error: {ekf_trace_error:.6f}")
print(f"  Relative Frobenius: {ekf_frobenius/np.linalg.norm(true_cov, 'fro'):.4f}")

# UKF errors
ukf_error = true_cov - ukf_cov
ukf_frobenius = np.linalg.norm(ukf_error, 'fro')
ukf_trace_error = abs(np.trace(true_cov) - np.trace(ukf_cov))

print("\nUKF Error (True - UKF):")
print(ukf_error)
print(f"  Frobenius norm: {ukf_frobenius:.6f}")
print(f"  Trace error: {ukf_trace_error:.6f}")
print(f"  Relative Frobenius: {ukf_frobenius/np.linalg.norm(true_cov, 'fro'):.4f}")

print("\n" + "="*70)
print("IMPROVEMENT")
print("="*70)
improvement = (ekf_frobenius - ukf_frobenius) / ekf_frobenius * 100
print(f"\nUKF is {improvement:.1f}% better than EKF (Frobenius norm)")
print(f"EKF Frobenius error: {ekf_frobenius:.6f}")
print(f"UKF Frobenius error: {ukf_frobenius:.6f}")

print("\n" + "="*70)
print("LEGEND COLORS")
print("="*70)
print("\n  🔵 Blue (solid):  True covariance from Monte Carlo")
print("  🟢 Green (solid): EKF linearized covariance")
print("  🟡 Yellow (solid): UKF covariance from sigma points")
print("  ⭐ Black star:    True mean point")
print("  ❌ Red X:         UKF sigma points")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
if ekf_frobenius > ukf_frobenius:
    print("\n✓ UKF provides more accurate covariance estimation!")
    print("  The sigma point approach better captures the nonlinear")
    print("  transformation from polar to Cartesian coordinates.")
    print("\n  Notice how the UKF sigma points (red X) spread around")
    print("  the true mean (black star) and better approximate the")
    print("  'banana shape' distribution compared to EKF's ellipse.")
else:
    print("\n✓ Both methods perform similarly for this scenario.")
    print("  The nonlinearity is not severe enough to show significant")
    print("  difference between Jacobian and sigma point methods.")

print("\n  The true mean (black star) shows the reference point,")
print("  while UKF sigma points show how UKF samples the uncertainty.")
print("="*70)