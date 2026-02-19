"""
distributed_decoder.py

Author: Kaan Okumus
Date: March 2025

This module implements the **distributed decoder** for TUMA in a **cell-free massive MIMO** setup. 
It decomposes centralized decoding into multiple **Access Points (APs)** that process signals independently 
before aggregation at the CPU.

### Key Components:
- **Access Point (`AP`) Class**:
  - Each AP runs **local AMP decoding** on its received signal.
  - Computes local channel and multiplicity estimates using **Bayesian denoising**.
- **CPU (`CPU`) Class**:
  - Aggregates log-likelihoods from all APs.
  - Performs a final **Bayesian estimation** step to compute global multiplicity estimates.
- **Distributed Decoding (`distributed_decoder`)**:
  - Executes the entire decoding pipeline across APs and the CPU.

### Main Functionalities:
- **`AP.AMP_decoder()`**: Runs AMP-based decoding at the AP level.
- **`CPU.aggregate_and_estimate_types()`**: Aggregates AP outputs to compute the final estimate.
- **`distributed_decoder()`**: Runs the full pipeline, including AP-level decoding and CPU aggregation.

### Usage:
- Compared against **centralized decoding (`centralized_decoder.py`)** in TUMA simulations.
- Used when **scalability** is preferred over optimal decoding performance.
"""

import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore

from bayesian_denoiser import *

class AP:
    def __init__(self, id, A, Cx, Cy, U, M, n, P, nAMPiter, Y, all_covs, all_covs_smaller, priors, log_priors, X_true=None, sigma_w=None, withOnsager=False):
        """
        Represents an Access Point (AP) in distributed decoding.

        Parameters:
        - id: AP index
        - A: Number of antennas per AP
        - Cx, Cy: Encoding and decoding functions
        - U: Number of zones
        - M: Number of messages per zone
        - n: Blocklength
        - P: Transmit power per user
        - nAMPiter: Number of AMP iterations
        - Y: Received signal matrix for this AP
        - all_covs, all_covs_smaller: Covariance matrices for denoising
        - priors, log_priors: Priors for Bayesian estimation
        - X_true: True transmitted signal (for performance comparison)
        - withOnsager: Whether to use Onsager correction in AMP
        """
        self.id = id
        self.A = A
        self.Cx = Cx
        self.Cy = Cy
        self.U = U
        self.M = M
        self.n = n
        self.P = P
        self.nP = n*P
        self.nAMPiter = nAMPiter
        self.all_covs = all_covs
        self.all_covs_smaller = all_covs_smaller
        self.priors = priors
        self.log_priors = log_priors
        self.Y = Y
        self.X_true = X_true
        self.sigma_w = sigma_w
        self.withOnsager = withOnsager

    def __str__(self):
        return f"AP {self.id+1}"

    def AMP_decoder(self):
        """
        Runs the AMP decoding algorithm at this AP.

        Steps:
        1. Initializes the residual signal `Z` and estimated signal `est_X`.
        2. Iterates through `nAMPiter` AMP iterations:
            - Computes residual noise covariance `T`.
            - Computes effective observation `R_u` for each zone.
            - Performs Bayesian denoising to estimate `est_X`.
            - Updates residual signal `Z` using Onsager correction if enabled.
        3. Stores local likelihoods for later aggregation by the CPU.
        """
        self.Z = self.Y.copy()
        #self.est_X_0 = [np.zeros((self.M,self.A), dtype=complex) for u in range(self.U)]
        self.est_X = [np.zeros((self.M,self.A), dtype=complex) for u in range(self.U)]

        self.tv_dists = []

        print(f"\t\tAP{self.id+1} AMP decoder ...")
        
        self.channel_est_perfs = []
        if self.X_true is not None:
            self.channel_est_perfs.append(np.sum([self.P * np.linalg.norm(self.X_true[u] - self.est_X[u], 'fro')**2 for u in range(self.U)]))
        self.channel_est_T_perfs = []
        
        for t in range(self.nAMPiter):
            print(f"\t\t\tAMPiter = {t+1}/{self.nAMPiter}")

            # Compute residual noise covariance
            taus = np.mean(np.diag(self.Z.conj().T @ self.Z).real/self.n)
            self.T = np.ones(self.A)*taus

            self.Gamma = np.zeros_like(self.Z)
            self.R = []
            for u in range(self.U):                
                # Compute effective observation for zone u
                R_u = self.Cy(self.Z, u) + np.sqrt(self.nP) * self.est_X[u]
                self.R.append(R_u)
                
                # Perform Bayesian estimation for each message
                Qu = np.zeros((self.A, self.A), dtype=complex)
                for m in range(self.M):
                    if self.withOnsager:
                        self.est_X[u][m], _, Qum = bayesian_channel_multiplicity_estimation_sampbasedapprox_with_onsager(
                            R_u[m], self.all_covs_smaller[u], self.T, self.nP, self.log_priors[u][m]
                        )
                        Qu += Qum                 
                    else:
                        self.est_X[u][m] = bayesian_channel_multiplicity_estimation_sampbasedapprox(R_u[m], self.all_covs_smaller[u], self.T, self.nP, self.log_priors[u][m])
                    
                # Update residual signal
                self.Gamma += self.Cx(self.est_X[u], u)
                self.Gamma -= (1 / self.n) * self.Z @ Qu
            self.Z = self.Y - np.sqrt(self.nP) * self.Gamma

            if self.X_true is not None:
                self.channel_est_perfs.append(np.sum([self.P * np.linalg.norm(self.X_true[u] - self.est_X[u], 'fro')**2 for u in range(self.U)]))
            if self.sigma_w is not None:
                self.channel_est_T_perfs.append(np.sum(np.abs(self.T - (self.sigma_w**2))))

        # Compute log-likelihoods for type estimation
        self.log_likelihoods = self.compute_local_likelihood(self.R, self.T, self.all_covs)
        
        if self.sigma_w is not None:
            taus = np.mean(np.diag(self.Z.conj().T @ self.Z).real/self.n)
            self.channel_est_T_perfs.append(np.sum(np.abs(np.ones(self.A)*taus - (self.sigma_w**2))))

    def normalize_posteriors(self, logposteriors):
        max_posterior = np.max(logposteriors)
        log_sumposteriors = max_posterior + np.log(np.exp(logposteriors - max_posterior).sum())
        return np.exp(logposteriors - log_sumposteriors)
    
    def compute_loglikelihoods_with_logsumexptrick(self, y, all_Covs, T, nP, Ns):
        logai = (-((np.abs(y)**2)/(T + nP * all_Covs) + np.log(np.pi * (T + nP * all_Covs))).sum(axis=-1) - np.log(Ns))
        maxlogai = np.max(logai,axis=-1, keepdims=True)
        loglikelihoods = (maxlogai[:,0] + np.log(np.exp(logai - maxlogai).sum(axis=-1)))
        return loglikelihoods

    def compute_local_likelihood(self, R, T, all_Covs):
        loglikelihoods = []
        Kmax = all_Covs.shape[1]
        Ns = all_Covs.shape[-2]
        for u in range(self.U):
            loglikelihoods_u = np.zeros((self.M, Kmax))
            for mu in range(self.M):
                loglikelihoods_um = self.compute_loglikelihoods_with_logsumexptrick(R[u][mu], all_Covs[u], T, self.nP, Ns)
                loglikelihoods_u[mu] = loglikelihoods_um
            loglikelihoods.append(loglikelihoods_u)
        return loglikelihoods

class CPU:
    def __init__(self, Y, U, A, B, all_Covs, all_Covs_smaller, priors, log_priors, Cx, Cy, M, n, P, nAMPiter, Xs_true, sigma_w=None, withOnsager=False):
        """
        Represents the central processing unit (CPU) that aggregates results from multiple APs.

        Parameters:
        - Y: Received signal
        - U: Number of zones
        - A: Number of antennas per AP
        - B: Number of APs
        - all_Covs, all_Covs_smaller: Covariance matrices
        - priors, log_priors: Priors for Bayesian estimation
        - Cx, Cy: Encoding and decoding functions
        - M: Number of messages per zone
        - n: Blocklength
        - P: Transmit power
        - nAMPiter: Number of AMP iterations
        - Xs_true: True transmitted signals (for performance comparison)
        - withOnsager: Whether to use Onsager correction
        """
        self.A = A
        self.B = B
        self.F = A*B
        self.Y = Y
        self.U = U
        self.log_priors = log_priors
        self.APs = []
        self.M = M

        # Initialize Access Points (APs)
        for b in range(B):
            self.APs.append(
                AP(id=b, A=A, Cx=Cx, Cy=Cy, U=U, M=M, n=n, P=P, 
                   nAMPiter=nAMPiter, Y=Y[:,b*A:(b+1)*A], all_covs=all_Covs[:,:,:,b*A:(b+1)*A], 
                   all_covs_smaller=all_Covs_smaller[:,:,:,b*A:(b+1)*A], priors=priors,
                   log_priors=log_priors, X_true=Xs_true[b], sigma_w=sigma_w, withOnsager=withOnsager)
            )

    def AMP_decoder(self):
        """
        Runs AMP decoding at each AP and stores estimated signals.
        """
        self.est_Xs = []
        self.channel_est_perfs = []
        self.channel_est_T_perfs = []
        for b in range(self.B):
            self.APs[b].AMP_decoder()
            self.est_Xs.append(self.APs[b].est_X)
            self.channel_est_perfs.append(self.APs[b].channel_est_perfs)
            self.channel_est_T_perfs.append(self.APs[b].channel_est_T_perfs)

    def compute_local_loglikelihoods(self):
        """
        Computes local likelihoods from all APs.
        """
        self.log_likelihoods_across_users = []
        for b in range(self.B):
            self.log_likelihoods_across_users.append(self.APs[b].log_likelihoods)
        self.log_likelihoods_across_users = np.array(self.log_likelihoods_across_users)

    def aggregate_and_estimate_types(self):
        """
        Aggregates likelihoods from all APs and estimates final message types.
        """
        self.log_posteriors = np.sum(self.log_likelihoods_across_users, axis=0) + self.log_priors
        self.est_k = np.zeros(self.U*self.M)
        for u in range(self.U):
            for m in range(self.M):
                self.est_k[u*self.M + m] = np.argmax(self.log_posteriors[u,m])

def distributed_decoder(Y, M, U, nAMPIter, B, A, Cx, Cy, nP, priors, log_priors, all_Covs, all_Covs_smaller, withOnsager=False, k_true=None, X_true=None, sigma_w=None, plot_perf=False):
    """
    Distributed decoder using multiple Access Points (APs) and a central CPU.

    Steps:
    1. Splits received signal `Y` among APs.
    2. Runs AMP decoding at each AP.
    3. Aggregates likelihoods across APs at the CPU.
    4. Estimates final multiplicities.
    """
    n, _ = Y.shape
    P = nP/n

    X_true_across_APs = []
    for b in range(B):
        sub_X = []
        for u in range(U):
            sub_X.append(X_true[u].reshape(-1, B, A)[:,b,:])
        X_true_across_APs.append(sub_X)

    cpu = CPU(Y=Y, U=U, A=A, B=B, all_Covs=all_Covs, all_Covs_smaller=all_Covs_smaller, priors=priors, log_priors=log_priors, Cx=Cx, Cy=Cy, M=M, n=n, P=P, nAMPiter=nAMPIter, Xs_true=X_true_across_APs, sigma_w=sigma_w, withOnsager=withOnsager)
    cpu.AMP_decoder()

    channel_est_perfs = np.array(cpu.channel_est_perfs).sum(axis=0)
    channel_est_T_perfs = np.array(cpu.channel_est_T_perfs).sum(axis=0)

    if plot_perf:
        for idx_AMP in range(len(channel_est_perfs)):
            print(f"\t{idx_AMP}\t{channel_est_perfs[idx_AMP]}\t \\\\")
        for idx_AMP in range(len(channel_est_T_perfs)):
            print(f"\t{idx_AMP}\t{channel_est_T_perfs[idx_AMP]}\t \\\\")
        plt.figure()
        plt.semilogy(channel_est_perfs, label="est_ch")
        plt.semilogy(channel_est_T_perfs, label="est_T")
        plt.legend()
        plt.show()

    cpu.compute_local_loglikelihoods()
    cpu.aggregate_and_estimate_types()

    est_k_per_zone = {}
    for u in range(U):
        est_k_per_zone[u] = cpu.est_k.reshape(-1,M)[u]

    return cpu.est_k.reshape(-1,M).sum(axis=0), est_k_per_zone
