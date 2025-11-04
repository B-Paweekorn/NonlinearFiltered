import numpy as np
import matplotlib.pyplot as plt
from NonlinearFiltered import ExtendedKalmanFilter, UnscentedKalmanFilter

# Time parameters
DT = 0.01  # Δt = 1 second
DURATION = 35.0  # 35 measurements

# Process noise
SIGMA_A = 1  # σ_a = 0.2 m/s²

# Measurement noise
SIGMA_RANGE = 5.0  # σ_rm = 5 m
SIGMA_BEARING = 0.01  # σ_φm = 0.0087 rad (~0.5 degrees)

# Initial estimate - rough initialization
INITIAL_STATE = np.array([20, 6, 0.0, 60, 1, 0.0])
INITIAL_COV = np.diag([50.0, 50.0, 50.0, 50.0, 50.0, 50.0])

# x vx ax y vy ay
INITIAL_TRAJ = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# ============================================================================
# TRUE TRAJECTORY GENERATION
# ============================================================================

def generate_true_trajectory(duration=DURATION, dt=DT):
    timestamps = np.arange(0, duration + dt, dt)
    n_steps = len(timestamps)
    
    # Pre-allocate array
    true_states = np.zeros((n_steps, 6))
    
    # Initial conditions
    x, y = 0.0, 0.0
    vx, vy = 0.0, 0.0
    
    for i, t in enumerate(timestamps):
        if t <= DURATION/5:
            ax = 5.0
            ay = 5.0
        elif DURATION/5 < t < DURATION/2:
            ax = 7.0
            ay = -9.0
        else:
            ax = -18.0
            ay = 6.0
        # Store current state
        true_states[i] = [x, vx, ax, y, vy, ay]
        
        # Update for next iteration
        vx += ax * dt
        vy += ay * dt
        x += vx * dt + 0.5*ax*dt*dt
        y += vy * dt + 0.5*ay*dt*dt
    
    return true_states, timestamps


def generate_measurements(true_states, sigma_range=SIGMA_RANGE, sigma_bearing=SIGMA_BEARING, seed=None):
    """
    Generate noisy radar measurements from TRUE states with NON-GAUSSIAN noise
    
    Measurement model: z = h(x) + v, where v is NON-GAUSSIAN
    h(x) = [sqrt(x² + y²), atan2(y, x)]
    
    Non-Gaussian noise types:
    - Range: Mixture of Gaussian + Uniform (heavy-tailed with outliers)
    - Bearing: Laplacian distribution (sharper peak, heavier tails than Gaussian)
    
    Args:
        true_states: True states (N, 6)
        sigma_range: Range noise scale (m)
        sigma_bearing: Bearing noise scale (rad)
        seed: Random seed for reproducibility
        
    Returns:
        measurements: (N, 2) array [range, bearing]
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_steps = len(true_states)
    measurements = np.zeros((n_steps, 2))
    
    for i in range(n_steps):
        x_pos = true_states[i, 0]
        y_pos = true_states[i, 3]
        
        # TRUE measurement (noiseless)
        true_range = np.sqrt(x_pos**2 + y_pos**2)
        true_bearing = np.arctan2(y_pos, x_pos)
        
        # NON-GAUSSIAN NOISE for range
        # Mixture: 70% Gaussian + 30% Uniform (creates heavy tails and outliers)
        if np.random.rand() < 0.9:
            # Gaussian component
            range_noise = np.random.randn() * sigma_range
        else:
            # Uniform component (outliers)
            range_noise = 15 + np.random.uniform(-6*sigma_range, 2*sigma_range)
        
        # NON-GAUSSIAN NOISE for bearing
        # Laplacian distribution (double exponential) - sharper peak, heavier tails
        bearing_noise = np.random.laplace(0, sigma_bearing / np.sqrt(2))
        
        # Alternative: Cauchy distribution (very heavy tails - uncomment to use)
        # bearing_noise = np.random.standard_cauchy() * sigma_bearing * 0.5
        
        # Add noise
        noisy_range = true_range + range_noise
        noisy_bearing = true_bearing + bearing_noise
        
        measurements[i] = [noisy_range, noisy_bearing]
    
    return measurements


# ============================================================================
# SYSTEM MODEL DEFINITIONS
# ============================================================================

class VehicleDynamics:
    """Vehicle dynamics model - 6D state"""
    
    def __init__(self, dt=DT, sigma_a=SIGMA_A):
        self.dt = dt
        self.sigma_a = sigma_a
        
    def get_F(self):
        """State transition matrix (Page 258)"""
        dt = self.dt
        return np.array([
            [1, dt, 0.5*dt**2, 0, 0, 0],
            [0, 1, dt, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, dt, 0.5*dt**2],
            [0, 0, 0, 0, 1, dt],
            [0, 0, 0, 0, 0, 1]
        ])
    
    def state_transition_function(self, state):
        """For UKF sigma point propagation"""
        return self.get_F() @ state
    
    def get_Q(self):
        """Process noise covariance (Page 258)"""
        dt = self.dt
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        
        Q = np.array([
            [dt4/4, dt3/2, dt2/2, 0, 0, 0],
            [dt3/2, dt2, dt, 0, 0, 0],
            [dt2/2, dt, 1, 0, 0, 0],
            [0, 0, 0, dt4/4, dt3/2, dt2/2],
            [0, 0, 0, dt3/2, dt2, dt],
            [0, 0, 0, dt2/2, dt, 1]
        ]) * (self.sigma_a ** 2)
        
        return Q


class RadarMeasurementModel:
    """Radar measurement model"""
    
    @staticmethod
    def h(state):
        """Measurement function h(x)"""
        x_pos = state[0]
        y_pos = state[3]
        
        range_val = np.sqrt(x_pos**2 + y_pos**2)
        bearing = np.arctan2(y_pos, x_pos)
        
        return np.array([range_val, bearing])
    
    @staticmethod
    def compute_jacobian(state):
        """Jacobian H = ∂h/∂x"""
        x_pos = state[0]
        y_pos = state[3]
        
        rho = np.sqrt(x_pos**2 + y_pos**2)
        
        if rho < 1e-6:
            rho = 1e-6
        
        rho_sq = rho * rho
        
        H = np.array([
            [x_pos/rho, 0, 0, y_pos/rho, 0, 0],
            [-y_pos/rho_sq, 0, 0, x_pos/rho_sq, 0, 0]
        ])
        
        return H
    
    @staticmethod
    def get_R():
        """Measurement covariance"""
        return np.diag([SIGMA_RANGE**2, SIGMA_BEARING**2])


def normalize_angle(angle):
    """Normalize angle to [-π, π]"""
    while angle > np.pi:
        angle -= 2.0 * np.pi
    while angle < -np.pi:
        angle += 2.0 * np.pi
    return angle


# ============================================================================
# FILTER IMPLEMENTATIONS
# ============================================================================

def run_ekf(measurements, initial_state, initial_cov):
    """Run EKF"""
    dynamics = VehicleDynamics()
    radar = RadarMeasurementModel()
    
    ekf = ExtendedKalmanFilter(state_dim=6)
    ekf.initialize(initial_state, initial_cov)
    
    n_steps = len(measurements)
    states = np.zeros((n_steps, 6))
    covariances = []
    
    states[0] = ekf.x
    covariances.append(ekf.P.copy())
    
    F = dynamics.get_F()
    Q = dynamics.get_Q()
    R = radar.get_R()
    
    for i in range(1, n_steps):
        # Predict
        ekf.predict(F, Q)
        
        # Update
        z_pred = radar.h(ekf.x)
        innovation = measurements[i] - z_pred
        innovation[1] = normalize_angle(innovation[1])
        
        H = radar.compute_jacobian(ekf.x)
        ekf.update(innovation, H, R)
        
        states[i] = ekf.x
        covariances.append(ekf.P.copy())
    
    return states, covariances


def run_ukf(measurements, initial_state, initial_cov):
    """Run UKF"""
    dynamics = VehicleDynamics()
    radar = RadarMeasurementModel()
    
    ukf = UnscentedKalmanFilter(state_dim=6, alpha=0.001, beta=2.0, kappa=0.0)
    ukf.initialize(initial_state, initial_cov)
    
    n_steps = len(measurements)
    states = np.zeros((n_steps, 6))
    covariances = []
    
    states[0] = ukf.x
    covariances.append(ukf.P.copy())
    
    Q = dynamics.get_Q()
    R = radar.get_R()
    
    for i in range(1, n_steps):
        # Predict
        ukf.predict(dynamics.state_transition_function, Q)
        
        # Update
        ukf.update(measurements[i], radar.h, R, angle_indices=[1])
        
        states[i] = ukf.x
        covariances.append(ukf.P.copy())
    
    return states, covariances


# ============================================================================
# MAIN SIMULATION
# ============================================================================

def run_simulation(seed=42):
    """
    Complete simulation: TRUE → Measurements → Estimates
    
    Args:
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with all results
    """
    print(f"\n{'='*70}")
    print("COMPLETE KALMAN FILTER SIMULATION")
    print(f"{'='*70}\n")
    
    print("Parameters (from book pages 258-259):")
    print(f"  Δt = {DT} s")
    print(f"  Duration = {DURATION} s")
    print(f"  σ_a = {SIGMA_A} m/s²")
    print(f"  σ_range = {SIGMA_RANGE} m")
    print(f"  σ_bearing = {SIGMA_BEARING:.4f} rad (~{np.rad2deg(SIGMA_BEARING):.2f}°)")
    
    # Step 1: Generate TRUE trajectory
    print(f"\nStep 1: Generating TRUE trajectory with piecewise constant acceleration...")
    true_states, timestamps = generate_true_trajectory(DURATION, DT)
    print(f"  → Generated {len(true_states)} states over {DURATION}s")
    print(f"  → Start: ({true_states[0,0]:.1f}, {true_states[0,3]:.1f})m | End: ({true_states[-1,0]:.1f}, {true_states[-1,3]:.1f})m")
    
    # Step 2: Generate measurements from TRUE + NON-GAUSSIAN noise
    print(f"\nStep 2: Generating noisy measurements with NON-GAUSSIAN noise...")
    measurements = generate_measurements(true_states, SIGMA_RANGE, SIGMA_BEARING, seed)
    
    # Calculate measurement errors
    true_measurements = np.array([RadarMeasurementModel.h(s) for s in true_states])
    meas_range_errors = measurements[:, 0] - true_measurements[:, 0]
    meas_bearing_errors = measurements[:, 1] - true_measurements[:, 1]
    
    print(f"  → Generated {len(measurements)} measurements")
    print(f"  → Range noise: Gaussian-Uniform mixture (70%-30%)")
    print(f"  → Bearing noise: Laplacian distribution")
    print(f"  → Range error: μ={np.mean(meas_range_errors):.2f}m, σ={np.std(meas_range_errors):.2f}m")
    print(f"  → Bearing error: μ={np.rad2deg(np.mean(meas_bearing_errors)):.3f}°, σ={np.rad2deg(np.std(meas_bearing_errors)):.3f}°")
    
    # Step 3: Initialize filters (rough initialization)
    print(f"\nStep 3: Initializing filters...")
    initial_error = np.linalg.norm(true_states[0, [0, 3]] - INITIAL_STATE[[0, 3]])
    print(f"  → Initial estimate: ({INITIAL_STATE[0]:.1f}, {INITIAL_STATE[3]:.1f})m | True: ({true_states[0,0]:.1f}, {true_states[0,3]:.1f})m")
    print(f"  → Initial error: {initial_error:.1f}m | Covariance: diag([50]×6)")
    
    # Step 4: Run EKF
    print(f"\nStep 4: Running Extended Kalman Filter...")
    ekf_states, ekf_covs = run_ekf(measurements, INITIAL_STATE, INITIAL_COV)
    print(f"  ✓ Completed {len(ekf_states)} iterations")
    
    # Step 5: Run UKF
    print(f"\nStep 5: Running Unscented Kalman Filter...")
    ukf_states, ukf_covs = run_ukf(measurements, INITIAL_STATE, INITIAL_COV)
    print(f"  ✓ Completed {len(ukf_states)} iterations")
    
    # Step 6: Calculate errors
    print(f"\nStep 6: Calculating estimation errors...")
    
    ekf_pos_errors = np.sqrt((true_states[:, 0] - ekf_states[:, 0])**2 + 
                              (true_states[:, 3] - ekf_states[:, 3])**2)
    ukf_pos_errors = np.sqrt((true_states[:, 0] - ukf_states[:, 0])**2 + 
                              (true_states[:, 3] - ukf_states[:, 3])**2)
    
    ekf_vel_errors = np.sqrt((true_states[:, 1] - ekf_states[:, 1])**2 + 
                              (true_states[:, 4] - ekf_states[:, 4])**2)
    ukf_vel_errors = np.sqrt((true_states[:, 1] - ukf_states[:, 1])**2 + 
                              (true_states[:, 4] - ukf_states[:, 4])**2)
    
    ekf_rmse = np.sqrt(np.mean(ekf_pos_errors**2))
    ukf_rmse = np.sqrt(np.mean(ukf_pos_errors**2))
    
    ekf_vel_rmse = np.sqrt(np.mean(ekf_vel_errors**2))
    ukf_vel_rmse = np.sqrt(np.mean(ukf_vel_errors**2))
    
    # Print results
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}\n")
    
    print("Position Estimation:")
    print(f"  EKF RMSE:        {ekf_rmse:.2f} m")
    print(f"  UKF RMSE:        {ukf_rmse:.2f} m")
    print(f"  Improvement:     {ekf_rmse - ukf_rmse:.2f} m ({100*(ekf_rmse-ukf_rmse)/ekf_rmse:.1f}%)")
    print(f"\n  EKF Final Error: {ekf_pos_errors[-1]:.2f} m")
    print(f"  UKF Final Error: {ukf_pos_errors[-1]:.2f} m")
    
    print(f"\nVelocity Estimation:")
    print(f"  EKF RMSE:        {ekf_vel_rmse:.2f} m/s")
    print(f"  UKF RMSE:        {ukf_vel_rmse:.2f} m/s")
    
    print(f"\nFinal Uncertainty (1σ):")
    print(f"  EKF: σ_x={np.sqrt(ekf_covs[-1][0,0]):.2f}m, σ_y={np.sqrt(ekf_covs[-1][3,3]):.2f}m")
    print(f"  UKF: σ_x={np.sqrt(ukf_covs[-1][0,0]):.2f}m, σ_y={np.sqrt(ukf_covs[-1][3,3]):.2f}m")
    
    print(f"\nConvergence:")
    print(f"  Initial error: {ekf_pos_errors[0]:.1f}m → Final: {ekf_pos_errors[-1]:.1f}m (EKF)")
    print(f"  Initial error: {ukf_pos_errors[0]:.1f}m → Final: {ukf_pos_errors[-1]:.1f}m (UKF)")
    
    return {
        'timestamps': timestamps,
        'true_states': true_states,
        'measurements': measurements,
        'ekf_states': ekf_states,
        'ukf_states': ukf_states,
        'ekf_covs': ekf_covs,
        'ukf_covs': ukf_covs,
        'ekf_pos_errors': ekf_pos_errors,
        'ukf_pos_errors': ukf_pos_errors,
        'ekf_rmse': ekf_rmse,
        'ukf_rmse': ukf_rmse
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(results):
    """Create comprehensive visualization"""
    
    timestamps = results['timestamps']
    true_states = results['true_states']
    measurements = results['measurements']
    ekf_states = results['ekf_states']
    ukf_states = results['ukf_states']
    ekf_pos_errors = results['ekf_pos_errors']
    ukf_pos_errors = results['ukf_pos_errors']
    ekf_covs = results['ekf_covs']
    ukf_covs = results['ukf_covs']
    
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Trajectory (X-Y)
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(true_states[:, 0], true_states[:, 3], 'g-', linewidth=3, label='True', zorder=3)
    ax1.plot(ekf_states[:, 0], ekf_states[:, 3], 'b--', linewidth=2, label='EKF', alpha=0.7)
    ax1.plot(ukf_states[:, 0], ukf_states[:, 3], 'r:', linewidth=2, label='UKF', alpha=0.7)
    ax1.plot(true_states[0, 0], true_states[0, 3], 'go', markersize=12, label='Start')
    ax1.plot(ekf_states[0, 0], ekf_states[0, 3], 'ko', markersize=10, label='Init Est.')
    ax1.set_xlabel('X Position (m)', fontweight='bold')
    ax1.set_ylabel('Y Position (m)', fontweight='bold')
    ax1.set_title('Vehicle Trajectory', fontweight='bold', fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # 2. Position Error
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(timestamps, ekf_pos_errors, 'b-', linewidth=2, label=f'EKF (RMSE={results["ekf_rmse"]:.1f}m)')
    ax2.plot(timestamps, ukf_pos_errors, 'r-', linewidth=2, label=f'UKF (RMSE={results["ukf_rmse"]:.1f}m)')
    ax2.set_xlabel('Time (s)', fontweight='bold')
    ax2.set_ylabel('Position Error (m)', fontweight='bold')
    ax2.set_title('Position Error Over Time', fontweight='bold', fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. X Position
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(timestamps, true_states[:, 0], 'g-', linewidth=3, label='True')
    ax3.plot(timestamps, ekf_states[:, 0], 'b--', linewidth=2, label='EKF', alpha=0.7)
    ax3.plot(timestamps, ukf_states[:, 0], 'r:', linewidth=2, label='UKF', alpha=0.7)
    ax3.set_xlabel('Time (s)', fontweight='bold')
    ax3.set_ylabel('X Position (m)', fontweight='bold')
    ax3.set_title('X Position', fontweight='bold', fontsize=13)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Y Position
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(timestamps, true_states[:, 3], 'g-', linewidth=3, label='True')
    ax4.plot(timestamps, ekf_states[:, 3], 'b--', linewidth=2, label='EKF', alpha=0.7)
    ax4.plot(timestamps, ukf_states[:, 3], 'r:', linewidth=2, label='UKF', alpha=0.7)
    ax4.set_xlabel('Time (s)', fontweight='bold')
    ax4.set_ylabel('Y Position (m)', fontweight='bold')
    ax4.set_title('Y Position', fontweight='bold', fontsize=13)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. X Velocity
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(timestamps, true_states[:, 1], 'g-', linewidth=3, label='True')
    ax5.plot(timestamps, ekf_states[:, 1], 'b--', linewidth=2, label='EKF', alpha=0.7)
    ax5.plot(timestamps, ukf_states[:, 1], 'r:', linewidth=2, label='UKF', alpha=0.7)
    ax5.set_xlabel('Time (s)', fontweight='bold')
    ax5.set_ylabel('X Velocity (m/s)', fontweight='bold')
    ax5.set_title('X Velocity', fontweight='bold', fontsize=13)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Y Velocity
    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(timestamps, true_states[:, 4], 'g-', linewidth=3, label='True')
    ax6.plot(timestamps, ekf_states[:, 4], 'b--', linewidth=2, label='EKF', alpha=0.7)
    ax6.plot(timestamps, ukf_states[:, 4], 'r:', linewidth=2, label='UKF', alpha=0.7)
    ax6.set_xlabel('Time (s)', fontweight='bold')
    ax6.set_ylabel('Y Velocity (m/s)', fontweight='bold')
    ax6.set_title('Y Velocity', fontweight='bold', fontsize=13)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Range Measurements
    ax7 = plt.subplot(3, 3, 7)
    true_range = np.sqrt(true_states[:, 0]**2 + true_states[:, 3]**2)
    ax7.plot(timestamps, true_range, 'g-', linewidth=2, label='True')
    ax7.scatter(timestamps, measurements[:, 0], c='orange', s=20, alpha=0.5, label='Measured')
    ax7.set_xlabel('Time (s)', fontweight='bold')
    ax7.set_ylabel('Range (m)', fontweight='bold')
    ax7.set_title('Radar Range (Non-Gaussian Noise)', fontweight='bold', fontsize=13)
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Bearing Measurements
    ax8 = plt.subplot(3, 3, 8)
    true_bearing = np.arctan2(true_states[:, 3], true_states[:, 0])
    ax8.plot(timestamps, np.rad2deg(true_bearing), 'g-', linewidth=2, label='True')
    ax8.scatter(timestamps, np.rad2deg(measurements[:, 1]), c='orange', s=20, alpha=0.5, label='Measured')
    ax8.set_xlabel('Time (s)', fontweight='bold')
    ax8.set_ylabel('Bearing (deg)', fontweight='bold')
    ax8.set_title('Radar Bearing (Laplacian Noise)', fontweight='bold', fontsize=13)
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Uncertainty Convergence
    ax9 = plt.subplot(3, 3, 9)
    ekf_std = [np.sqrt(P[0,0]) for P in ekf_covs]
    ukf_std = [np.sqrt(P[0,0]) for P in ukf_covs]
    ax9.plot(timestamps, ekf_std, 'b-', linewidth=2, label='EKF σ_x')
    ax9.plot(timestamps, ukf_std, 'r-', linewidth=2, label='UKF σ_x')
    ax9.set_xlabel('Time (s)', fontweight='bold')
    ax9.set_ylabel('Uncertainty (m)', fontweight='bold')
    ax9.set_title('Position Uncertainty', fontweight='bold', fontsize=13)
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    ax9.set_yscale('log')
    
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    # Run complete simulation
    results = run_simulation(seed=42)
    
    # Plot
    plot_results(results)
    
    print("\n" + "="*70)
    print("Simulation complete! Close plot window to exit.")
    print("="*70 + "\n")
    
    plt.show()