# EKF vs UKF Comparative Analysis

Comparative study of Extended Kalman Filter (EKF) and Unscented Kalman Filter (UKF) for radar-based vehicle tracking with non-Gaussian noise.

## Overview

This repository implements and compares two nonlinear state estimation algorithms:
- **Extended Kalman Filter (EKF)**: Uses Jacobian linearization
- **Unscented Kalman Filter (UKF)**: Uses sigma point unscented transform

The comparison focuses on radar tracking scenarios where measurements arrive in polar coordinates (range, bearing) but estimation is performed in Cartesian space.

## Key Results

- **UKF achieves 10.6% better position RMSE** compared to EKF
- **93.7% more accurate covariance propagation** for polar-to-Cartesian transformations
- **33% faster convergence** from poor initialization
- Superior robustness to non-Gaussian measurement noise

## Repository Structure

```
.
├── NonlinearFiltered.py    # Generic EKF and UKF implementations
├── scenarios.py            # Vehicle tracking simulation with non-Gaussian noise
├── cov.py                  # Covariance propagation comparison
└── README.md
```

## Files Description

### `NonlinearFiltered.py`
Generic filter implementations that can be adapted to any nonlinear system:
- `ExtendedKalmanFilter`: Jacobian-based linearization approach
- `UnscentedKalmanFilter`: Sigma point-based approach with robust Cholesky decomposition

### `scenarios.py`
Complete radar tracking simulation:
- 6D state space: position, velocity, acceleration (x, y)
- Piecewise constant acceleration trajectory (3 maneuver phases)
- Non-Gaussian measurement noise:
  - Range: Gaussian-Uniform mixture (90%/10%)
  - Bearing: Laplacian distribution
- Comprehensive visualization (9 subplots)

### `cov.py`
Demonstrates fundamental difference between EKF and UKF:
- Polar-to-Cartesian coordinate transformation
- Monte Carlo validation (1000 samples)
- Visual comparison of covariance ellipses
- Quantitative error metrics

## Usage

### Vehicle Tracking Simulation

```bash
python scenarios.py
```

**Output:**
- Console: Step-by-step progress, final RMSE comparison
- Plots: Trajectory, errors, position/velocity tracking, measurements, uncertainty convergence

### Covariance Propagation Analysis

```bash
python cov.py
```

**Output:**
- Side-by-side comparison: polar coordinates → Cartesian coordinates
- True covariance (Monte Carlo) vs EKF (green) vs UKF (red)
- Frobenius norm error comparison

## Requirements

```
numpy
matplotlib
scipy
```

Install dependencies:
```bash
pip install numpy matplotlib scipy
```

## Key Parameters

From `scenarios.py`:

```python
DT = 0.01                    # Time step (s)
DURATION = 35.0              # Simulation duration (s)
SIGMA_A = 1                  # Process noise (m/s²)
SIGMA_RANGE = 5.0            # Range measurement noise (m)
SIGMA_BEARING = 0.01         # Bearing measurement noise (rad)

# Initial estimate (intentionally poor)
INITIAL_STATE = [200, 60, 0.0, 600, 10, 0.0]  # [x, vx, ax, y, vy, ay]
INITIAL_COV = diag([50, 50, 50, 50, 50, 50])

# UKF tuning
alpha = 0.001               # Sigma point spread
beta = 2.0                  # Gaussian distribution parameter
kappa = 0.0                 # Secondary scaling
```

## Theoretical Background

**EKF Approach:**
- Linearizes nonlinear functions using Jacobian matrices
- First-order Taylor series approximation
- Fast but can introduce significant errors

**UKF Approach:**
- Propagates carefully chosen sigma points through true nonlinear functions
- No derivatives required
- Second-order accuracy in mean, third-order in covariance

## Measurement Model

Radar provides range and bearing:

```
ρ = √(x² + y²) + noise
φ = arctan2(y, x) + noise
```

State transition follows constant acceleration dynamics with process noise modeling uncertainty in acceleration.

## Results Summary

| Metric | EKF | UKF | Improvement |
|--------|-----|-----|-------------|
| Position RMSE (m) | 14.1 | 12.6 | 10.6% |
| Final Error (m) | 8.2 | 5.4 | 34.1% |
| Convergence Time (s) | 15 | 10 | 33% faster |
| Covariance Accuracy | Baseline | 16× better | 93.7% |

## Citation

If you use this code in your research, please cite:

```
Buasakorn, P. (2024). Comparative Analysis of Extended and Unscented 
Kalman Filters for Radar-Based 2D Position Estimation. 
Xi'an Jiaotong University.
```

## Acknowledgments

This work builds upon concepts from:
- Alex Becker's "Kalman Filter from the Ground Up"
- Julier & Uhlmann's original UKF paper (1997)
- Course materials from Prof. Zhansheng Duan, XJTU

## Contact

Paweekorn Buasakorn  
Xi'an Jiaotong University  
paweekorn.pb@gmail.com
