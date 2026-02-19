"""
prior.py

Author: Kaan Okumus
Date: March 2025

This module computes the **prior probability distributions** for message multiplicities in TUMA, 
which are used for **Bayesian estimation** in the AMP decoding process.

### Key Components:
- **Detection Probability (`detection_probability`)**:
  - Computes the probability of detecting a target given a sensor position and system parameters.
  - Uses the **generalized Marcum Q-function** for modeling detection likelihood.
- **Sensor Activation Probability (`compute_P_active`)**:
  - Estimates the probability that a sensor detects at least one target.
  - Uses **Monte Carlo integration** for accurate probability estimation.
- **Message Selection Probability (`compute_P_m_given_zone_u`)**:
  - Computes the probability of a sensor selecting a message in a given zone.
  - Uses **spatial sampling and probabilistic modeling**.
- **Message Multiplicity Probability (`compute_P_k_um`)**:
  - Models message multiplicities using **binomial distributions**.
  - Computes `P(k_{u,m})`, the probability distribution of message occurrences in each zone.
- **P_closest Computation (`compute_P_closest`, `precompute_P_closest`)**:
  - Determines the probability that a detected target is the closest to a sensor.
  - Uses **Monte Carlo simulations** and interpolation for efficiency.

### Main Functionalities:
- **`detection_probability()`**: Computes the probability of a target being detected by a sensor.
- **`compute_P_active()`**: Determines the probability of a sensor detecting at least one target.
- **`compute_P_m_given_zone_u()`**: Computes the probability distribution of message selection per zone.
- **`compute_P_k_um()`**: Computes the probability distribution of message multiplicities.
- **`compute_prior()`**: Calls all necessary functions to compute prior distributions for TUMA.

### Usage:
- Used in **TUMAEnvironment (`tuma.py`)** before running the AMP decoder.
- Utilized in **centralized (`centralized_decoder.py`)** and **distributed (`distributed_decoder.py`)** decoders 
  for **Bayesian denoising and estimation**.
"""

import numpy as np # type: ignore
import scipy.stats as stats # type: ignore
from scipy.interpolate import RegularGridInterpolator # type: ignore

def marcumq(a, b, m=1):
    """
    Computes the generalized Marcum Q-function.
    """
    return stats.ncx2.sf(b**2, 2*m, a**2)
    
def detection_probability(s, t, Ns, Ps=0.0009765625, sigma_n=10**(-5), S=100, fc=2.8e9, gamma=2*np.log(1e8)):
    """
    Computes the probability of detecting a target at t from a sensor at s.
    """
    dist = np.linalg.norm(s - t, axis=-1)*1000  # Convert to meters
    dist = np.maximum(dist, 1e-6)  # Avoid zero distance issue
    lambda_ = 3e8 / fc  # Wavelength
    return marcumq(np.sqrt((2 * Ns * Ps * S * (lambda_**2)) / (((4 * np.pi)**3) * (dist**4) * (sigma_n**2))), np.sqrt(gamma))

def compute_P_active(T, N_s, zone_side, N_t=100, num_mc_samples=100):
    """
    Computes P_active, the probability that a sensor detects at least one target.
    Uses Monte Carlo integration with sensor-target detection modeling.
    """
    region_size = 3 * zone_side  # Total region size (3a x 3a)    
    sensor_positions = np.random.uniform(0, region_size, size=(num_mc_samples, 2))    
    target_positions = np.random.uniform(0, region_size, (N_t, T, 2))
    detection_probs = detection_probability(sensor_positions[:, None, None, :], target_positions[None, :, :, :], N_s)
    P_no_detection = np.mean(np.prod(1 - detection_probs, axis=2), axis=1)
    return 1 - np.mean(P_no_detection)

def compute_P_Kau(K, P_active, U):
    """
    Computes the probability mass function of K_au (active users per zone).
    Uses binomial distribution to model allocation of active users to zones.
    """
    Ka_values = np.arange(0, K + 1)
    Kau_values = np.arange(0, K + 1)

    P_Ka = stats.binom.pmf(Ka_values, K, P_active)
    P_Kau = np.array([np.sum([stats.binom.pmf(k_au, k_a, 1/U) * P_Ka[j] for j, k_a in enumerate(Ka_values)]) for k_au in Kau_values])

    return Kau_values, P_Kau

def compute_P_closest(s, t, region_size, T, N_s, N_mc_samples=500):
    """
    Computes P_closest(s,t), the probability that target t is the closest detected target to sensor s.
    Uses Monte Carlo estimation with vectorized operations.
    """
    # Sample additional targets randomly in the region (N_mc_samples, T-1, 2)
    target_samples = np.random.uniform(0, region_size, size=(N_mc_samples, T-1, 2))

    # Compute pairwise distances (N_mc_samples, T-1)
    distances = np.linalg.norm(target_samples - s, axis=-1)  
    ref_distance = np.linalg.norm(t - s)

    # Compute detection probability for each target
    detection_probs = detection_probability(s, target_samples, N_s)  # Shape: (N_mc_samples, T-1)

    # Determine if sampled target is closer than t
    closer_mask = (distances < ref_distance)  # Boolean array (N_mc_samples, T-1)

    # Compute the probability that none of the closer targets are detected
    P_no_closer_detection = np.prod(1 - closer_mask * detection_probs, axis=1)

    return np.mean(P_no_closer_detection)

def precompute_P_closest(region_size, T, N_s, grid_size=10, N_mc_samples=100):
    """
    Precomputes P_closest values over a grid for fast interpolation.
    """
    x_vals = np.linspace(0, region_size, grid_size)
    y_vals = np.linspace(0, region_size, grid_size)
    P_closest_values = np.zeros((grid_size, grid_size, grid_size, grid_size))

    for i, x_s in enumerate(x_vals):
        for j, y_s in enumerate(y_vals):
            s = np.array([x_s, y_s])
            for k, x_t in enumerate(x_vals):
                for l, y_t in enumerate(y_vals):
                    t = np.array([x_t, y_t])
                    P_closest_values[i, j, k, l] = compute_P_closest(s, t, region_size, T, N_s, N_mc_samples)

    return RegularGridInterpolator((x_vals, y_vals, x_vals, y_vals), P_closest_values)

def compute_P_m_given_zone_u(U, M, region_size, T, N_s, num_mc_samples=100, N_t=100, include_Pclosest=True):
    """
    Computes P(m | sensor in zone u) using vectorized operations to speed up execution.
    """
    P_m_given_zone_u = np.zeros((U, M))  # Store results for all zones and messages

    zone_centers = np.array([((i + 0.5) * (region_size / np.sqrt(U)), (j + 0.5) * (region_size / np.sqrt(U))) for j in range(int(np.sqrt(U))) for i in range(int(np.sqrt(U)))])
    message_centers = np.array([((i + 0.5) * (region_size / np.sqrt(M)), (j + 0.5) * (region_size / np.sqrt(M))) for j in range(int(np.sqrt(M))) for i in range(int(np.sqrt(M)))])

    cell_side = region_size / np.sqrt(M)

    if include_Pclosest:
        print("\t- Precomputing P_closest ...")
        P_closest_interpolator = precompute_P_closest(region_size, T, N_s)
    else:
        P_closest_interpolator = lambda _ : 1.0

    print("\t- Computing P_m_given_zone_u using vectorized operations ...")

    for u, zone_center in enumerate(zone_centers):
        print(f"\t\tu={u+1}/{U}")

        # Sample sensor positions from zone u (num_mc_samples x 2)
        sensor_positions = np.random.uniform(
            low=zone_center - (region_size / (2 * np.sqrt(U))),
            high=zone_center + (region_size / (2 * np.sqrt(U))),
            size=(num_mc_samples, 2)
        )

        for m, message_center in enumerate(message_centers):
            # Sample target positions inside R_m (N_t x 2)
            target_samples = np.random.uniform(
                message_center - cell_side / 2, 
                message_center + cell_side / 2, 
                size=(N_t, 2)
            )

            # Compute pairwise detection probabilities: (num_mc_samples, N_t)
            detection_probs = detection_probability(sensor_positions[:, None, :], target_samples[None, :, :], N_s)

            # Compute P_closest efficiently
            if include_Pclosest:
                P_closest_values = np.array([
                    P_closest_interpolator((s[0], s[1], t[0], t[1])) for t in target_samples for s in sensor_positions
                ]).reshape(num_mc_samples, N_t)
            else:
                P_closest_values = np.ones((num_mc_samples, N_t))  # If P_closest is ignored, assume it's 1

            # Compute P(m | sensor in zone u) using vectorized mean operations
            P_m_given_zone_u[u, m] = np.mean(P_closest_values * detection_probs) / U

        # Normalize to ensure sum_m P(m | sensor in zone u) = 1
        P_m_given_zone_u[u, :] /= np.maximum(np.sum(P_m_given_zone_u[u, :]), 1e-6)

    return P_m_given_zone_u

def compute_P_k_um(K, U, M, P_active, P_m_given_zone_u):
    """
    Computes P(k_{u,m}), the probability distribution of message multiplicities.
    Uses binomial distributions to model message occurrences.
    """
    P_k_um = np.zeros((U, M, K+1))
    K_au_values = np.arange(0, K+1)
    P_Kau = stats.binom.pmf(K_au_values[:, None], K, P_active / U)  

    for u in range(U):
        for m in range(M):
            Pm_u = P_m_given_zone_u[u, m]
            K_um_values = np.arange(0, K+1)
            P_Kum_given_Kau = stats.binom.pmf(K_um_values[:, None], K_au_values, Pm_u)
            P_k_um[u, m, :] = np.sum(P_Kum_given_Kau * P_Kau.T, axis=1)

    return P_k_um

def compute_prior(K, T, U, N_s, M, zone_side, include_Pclosest=True):
    """
    Computes the prior probability of message multiplicities for TUMA.
    Steps:
    1. Compute P_active (probability of a sensor getting activated).
    2. Compute P(m | zone u) for message selection in each zone.
    3. Compute P(k_{u,m}) for message multiplicities.
    """
    print("Prior computation ...")
    print("\t- Computing P_active ...")
    P_active = compute_P_active(T, N_s, zone_side)
    print("\t- Computing P_m_given_zone_u ...")
    P_m_given_zone_u = compute_P_m_given_zone_u(U, M, 3 * zone_side, T, N_s, include_Pclosest=include_Pclosest)
    print("\t- Computing P_k_um ...")
    prior = compute_P_k_um(K, U, M, P_active, P_m_given_zone_u)
    print("Prior computation complete.")
    return prior
