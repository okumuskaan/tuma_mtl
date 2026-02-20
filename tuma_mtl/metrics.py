"""
metrics.py

Author: Kaan Okumus  
Date: March 2025  

This module defines **performance metrics** for evaluating multi-target localization (MTL)  
and TUMA-based communication systems. It includes **total variation (TV) distance** and  
**truncated Wasserstein distance** for assessing the accuracy of estimated target positions  
and message multiplicities.

### Key Components:
- **True Multiplicities Computation**:
  - `get_true_positions_and_type()`: Extracts the true target positions and their multiplicities  
    from active sensors.
- **Evaluation Metrics**:
  - `tv_distance()`: Computes the total variation (TV) distance between true and estimated types.
  - `wasserstein_distance()`: Computes the truncated Wasserstein distance for target position estimation.

### Main Functionalities:
- **`get_true_positions_and_type(active_sensors)`**:
  - Extracts unique target positions from sensors.
  - Computes the multiplicities of each detected target.
  - Normalizes the multiplicities to derive the **true type distribution**.

- **`tv_distance(true_type, est_type)`**:
  - Measures the discrepancy between the true and estimated type distributions.
  - Defined as:  \[ TV = \frac{1}{2} \sum |p_i - q_i| \]
  - Values range from **0 (perfect match)** to **1 (completely different distributions)**.

- **`wasserstein_distance(F1, F2, W1, W2, c=np.inf, p=2)`**:
  - Computes the **truncated Wasserstein distance** between true and estimated target positions.
  - Uses a cost matrix to represent the transport problem between **F1 (true positions)**  
    and **F2 (estimated positions)**, weighted by their respective multiplicities.
  - Solves a **linear programming problem** to find the optimal transport.
  - **Truncation parameter (`c`)** ensures that very large distances do not dominate the cost.

### Usage:
This module is used for **evaluating TUMA-based MTL systems**, particularly to:
- Compare estimated and true target distributions.
- Assess the quality of target position estimation.
- Measure the impact of message multiplicity errors.
"""

import numpy as np # type: ignore
import scipy.optimize # type: ignore
from collections import defaultdict

def get_true_positions_and_type(active_sensors):
    """
    Extracts true target positions and their multiplicities from active sensors.

    Parameters:
    active_sensors : list
        List of active Sensor objects.

    Returns:
    tuple:
        - true_positions (np.ndarray): Unique detected target positions.
        - true_multiplicities (np.ndarray): Multiplicity counts for each detected target.
        - true_type (np.ndarray): Normalized type distribution of detected targets.
    """
    true_multiplicities = defaultdict(int)
    for sensor in active_sensors:
        for _, target_pos in sensor.detected_targets.items():
            true_multiplicities[target_pos] += 1  # Increment count for that target
    true_positions = np.array(list(true_multiplicities.keys()))  # Unique target positions
    true_multiplicities = np.array(list(true_multiplicities.values()))  # Multiplicity per target
    true_type = true_multiplicities/(true_multiplicities.sum()) # True type
    return true_positions, true_multiplicities, true_type

def tv_distance(true_type, est_type):
    """
    Computes the total variation (TV) distance between two type distributions.

    Parameters:
    true_type : np.ndarray
        True type distribution (normalized multiplicities).
    est_type : np.ndarray
        Estimated type distribution.

    Returns:
    float
        Total variation distance (ranges from 0 to 1).
    """
    return np.abs(true_type - est_type).sum()/2

def wasserstein_distance(F1, F2, W1, W2, c=np.inf, p=2):
    """
    Computes the truncated Wasserstein distance between true and estimated target positions.

    Parameters:
    F1 : np.ndarray, shape (m, d)
        Positions of true detected targets.
    F2 : np.ndarray, shape (n, d)
        Positions of estimated targets.
    W1 : np.ndarray, shape (m, 1)
        Multiplicities of true detected targets.
    W2 : np.ndarray, shape (n, 1)
        Multiplicities of estimated targets.
    c : float, optional
        Truncation parameter for large distances (default: np.inf, no truncation).
    p : int, optional
        Wasserstein distance exponent (default: 2).

    Returns:
    float
        Truncated Wasserstein distance.
    """
    def truncated_distance(x, y, c):
        """Computes the truncated p-norm distance between two points."""
        return min(np.linalg.norm(x - y), c)**p

    # Construct the cost matrix (m x n)
    f = np.zeros((len(F1), len(F2)))
    for i in range(len(F1)):
        for j in range(len(F2)):
            f[i, j] = truncated_distance(F1[i], F2[j], c)

    f = f.flatten()
    m, n = len(F1), len(F2)

    # Construct constraint matrices for transport optimization
    A1 = np.zeros((m, m * n)) # Row constraints
    A2 = np.zeros((n, m * n)) # Column constraints
    for i in range(m):
        for j in range(n):
            k = j + i * n
            A1[i, k] = 1
            A2[j, k] = 1

    # Stack constraints
    A = np.vstack((A1, A2))
    b = np.vstack((W1, W2))

    # Equality constraint to ensure transport balance
    Aeq = np.ones((1, m * n))
    beq = np.min([np.sum(W1), np.sum(W2)])

    # Solve the linear programming problem to find the optimal transport plan
    res = scipy.optimize.linprog(
        f,
        A_ub=None,
        b_ub=None,
        A_eq=np.vstack((Aeq, A)),
        b_eq=np.vstack((beq, b)),
        bounds=(0, None),
        method='highs'
    )

    # Compute the Wasserstein distance from the optimal transport cost
    fval = (res.fun / np.sum(res.x))**(1/p) if np.sum(res.x) > 0 else 0

    return fval

def dGOSPAlike(empirical_pmd, ws_dist, c, p=2):
    return (
        ws_dist**p +
        (empirical_pmd * c**p)
    )**(1/p)


def compute_tuma_mtl_performance_metric(true_positions, detected_positions, estimated_positions, 
                            detected_types, estimated_types, c, p=2):
    """
    Computes the TUMA-MTL performance metric.

    Parameters:
    - true_positions: np.array, shape (T, 1) - Positions of all true targets (active & inactive).
    - detected_positions: np.array, shape (T_d, 1) - Positions of detected true targets.
    - estimated_positions: np.array, shape (T̂_d, 1) - Positions of estimated active targets.
    - detected_types: np.array, shape (T_d, 1) - True multiplicities of detected targets.
    - estimated_types: np.array, shape (T̂_d, 1) - Estimated multiplicities of detected targets.
    - c: float - Truncation parameter for Wasserstein distance.
    - alpha: float - Parameter for missed detection penalty.
    - p: int - Wasserstein distance exponent.

    Returns:
    - empirical_pmd: int - Empirical misdetection probability.
    - ws_dist: float - Wasserstein distance between distributions.
    - dGOSPAlike: float - The computed GOSPA-like cost metric.
    """

    T = len(true_positions)  # Total true targets
    T_d = len(detected_positions)  # Detected targets

    # Compute Wasserstein distance for communication errors
    ws_dist = wasserstein_distance(
        detected_positions.reshape(-1, 1),
        estimated_positions.reshape(-1, 1),
        detected_types.reshape(-1, 1),
        estimated_types.reshape(-1, 1),
        p=p
    )

    # Compute empirical missed detection probability in sensing (true targets not detected)
    empirical_pmd = 1- T_d/T

    # Compute final GOSPA-like cost metric
    dGOSPAlike = (
        ws_dist**p +
        (empirical_pmd * c**p)
    )**(1/p)
    
    return empirical_pmd, ws_dist, dGOSPAlike