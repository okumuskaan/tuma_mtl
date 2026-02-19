"""
centralized_decoder.py

Author: Kaan Okumus
Date: March 2025

This module implements the **centralized decoder** for TUMA with fading in cell-free (CF) massive MIMO. 
It utilizes **Approximate Message Passing (AMP)** and **Bayesian denoising** to estimate transmitted messages.

### Key Components:
- **Centralized Decoding (`centralized_decoder`)**:
  - Processes received signal `Y` to recover transmitted message multiplicities.
  - Uses **sampling-based Bayesian estimation** for **channel and multiplicity estimation**.
  - Supports optional **Onsager correction** to improve AMP convergence.
- **Covariance Matrix Computation (`generate_all_covs`)**:
  - Precomputes covariance matrices for different user locations and multiplicities.
  - Improves efficiency in Monte Carlo (MC) simulations.
- **Residual Noise Estimation (`compute_T`)**:
  - Computes noise covariance matrix for iterative AMP updates.

### Main Functionalities:
- **`centralized_decoder()`**: Runs the AMP algorithm and Bayesian inference.
- **`generate_all_covs()`**: Generates covariance matrices for all possible multiplicities.
- **`compute_T()`**: Computes the covariance of the residual noise.

### Usage:
- Used within **TUMAEnvironment (`tuma.py`)** for decoding transmitted messages.
- Compared against the **distributed decoder (`distributed_decoder.py`)** to analyze performance.
"""

import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from copy import deepcopy

from bayesian_denoiser import *

def compute_T(Z, n, A, B):
    """ Computes covariance matrix of residual noise. """
    c = np.diag(Z.conj().T @ Z).real/n
    taus = np.mean(c.reshape(B,A),axis=1)
    taus = (np.ones((1,A))*(taus.reshape(-1,1))).reshape(-1)
    return taus

def centralized_decoder(Y, M, U, nAMPIter, B, A, Cx, Cy, nP, priors, log_priors, all_Covs, all_Covs_smaller, withOnsager=False, k_true=None, X_true=None, sigma_w=None, plot_perf=False):
    """
    Centralized Decoder with Multisource AMP and Bayesian Estimation.

    Parameters:
    Y : np.ndarray
        Received signal matrix.
    M : int
        Number of codewords.
    U : int
        Number of zones.
    nAMPIter : int
        Number of AMP iterations.
    B : int
        Number of access points (APs).
    A : int
        Number of antennas per AP.
    Cx, Cy : functions
        Encoding (C) and decoding (C^H) functions.
    nP : float
        Normalized power per transmission.
    priors : list[np.ndarray]
        Priors for each zone.
    all_Covs : np.ndarray
        Covariance matrices for sampling-based approximation (specifically used for multiplicity estimation).
    all_Covs_smaller : np.ndarray
        Smaller covariance matrices for sampling-based approximation (specifically used for channel estimation).
    withOnsager: bool
        Boolean variable for including Onsager term.
    k_true: np.ndarray
        True multiplicity vector.
    X_true: np.ndarray
        True channel matrix.
    sigma_w: float
        True AWGN noise standard deviation.
    plot_perf: bool
        Boolean variable for plotting the channel and multiplicity estimation performance vs AMP iteration.

    Returns:
    est_k : np.ndarray
        Estimated multiplicity vector.
    est_k_per_zone : list[np.ndarray]
        Estimated multiplicity vectors per zones. 
    """
    # Initialization
    n, F = Y.shape
    P = nP/n
    Z = Y.copy()  # residual signal
    est_X = [np.zeros((M, F), dtype=complex) for _ in range(U)] # estimated channel matrix
    tv_dists = []
    channel_est_perfs = []
    if X_true is not None:
        channel_est_perfs.append(P * np.sum([np.linalg.norm(X_true[u] - est_X[u], 'fro')**2 for u in range(U)]))
    channel_est_T_perfs = []

    prev_est_X = None
    for t in range(nAMPIter):
        print(f"\t\tAMP iteration: {t+1}/{nAMPIter}")
        
        # Residual Covariance
        T = compute_T(Z, n, A, B)

        # AMP Updates
        Gamma = np.zeros_like(Z)
        R = []
        for u in range(U):
            R_u = Cy(Z, u) + np.sqrt(nP) * est_X[u]
            R.append(R_u)
            Qu = np.zeros((F,F), dtype=complex)
            for m in range(M):

                if withOnsager:
                    est_X[u][m], est_k, Qum = bayesian_channel_multiplicity_estimation_sampbasedapprox_with_onsager(
                        R_u[m], all_Covs_smaller[u], T, nP, log_priors[u][m]
                    )
                    Qu += Qum
                else:
                    est_X[u][m] = bayesian_channel_multiplicity_estimation_sampbasedapprox(R_u[m], all_Covs_smaller[u], T, nP, log_priors[u][m])

            Gamma += Cx(est_X[u], u)
            Gamma -= (1/n) * Z @ Qu

        Z = Y - np.sqrt(nP) * Gamma

        # Estimate multiplicity/type
        est_k, posteriors = estimate_type_samplingbased_logsumexp(R, T, all_Covs, M, U, nP, log_priors)

        # Update priors
        damp=1.0
        priors = [damp*priors[u] + (1-damp)*posteriors[u] for u in range(U)]
        log_priors = [np.log(priors[u]) for u in range(U)]

        # TV Distance
        if k_true is not None:
            tv_dist = np.sum(np.abs(k_true / np.sum(k_true) - est_k / np.sum(est_k))) / 2
            tv_dists.append(tv_dist)

        if X_true is not None:
            channel_est_perfs.append(P * np.sum([np.linalg.norm(X_true[u] - est_X[u], 'fro')**2 for u in range(U)]))
        if sigma_w is not None:
            channel_est_T_perfs.append(np.sum(np.abs(T - (sigma_w**2))))

        # Convergence Check
        if t >= 3:
            channel_est_perf_diff = P * np.sum([np.linalg.norm(prev_est_X[u] - est_X[u], 'fro')**2 for u in range(U)])
            if channel_est_perf_diff<1e-7:
                break

        prev_est_X = deepcopy(est_X)

    if sigma_w is not None:
        channel_est_T_perfs.append(np.sum(np.abs(compute_T(Z, n, A, B) - (sigma_w**2))))

    if plot_perf:
        print("channel_est_perfs =", channel_est_perfs)
        print("channel_est_T_perfs =", channel_est_T_perfs)
        plt.figure()
        plt.semilogy(np.arange(0, len(channel_est_perfs)), channel_est_perfs, label="Channel Est")
        plt.semilogy(np.arange(0, len(channel_est_perfs)), channel_est_T_perfs, "--", label="Residual Cov Est")
        plt.legend()
        plt.grid("True")
        plt.xlabel("AMP iteration number")
        plt.ylabel("Multisource AMP performance scores")
        plt.show()

        plt.figure()
        plt.plot(np.arange(1, len(tv_dists)+1), tv_dists)
        plt.ylabel("TV distance")
        plt.xlabel("AMP iteration number")
        plt.grid("True")
        plt.show()


    est_k_per_zone = {}
    for u in range(U):
        est_k_per_zone[u] = est_k.reshape(-1,M)[u]

    return est_k.reshape(-1,M).sum(axis=0), est_k_per_zone
