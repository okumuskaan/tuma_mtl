"""
tuma.py

Author: Kaan Okumus
Date: March 2025

This module defines the **TUMA Environment**, which integrates **sensing, communication, and decoding** 
for **multi-target localization (MTL) and federated learning (FL)** applications.

### Key Components:
- **TUMA Environment (`TUMAEnvironment`)**:
  - Manages **zone assignment**, **codebook generation**, and **sensor preparation**.
  - Simulates **TUMA encoding and transmission**.
  - Runs either **centralized or distributed AMP decoding**.
- **Transmission and Noise Modeling**:
  - Implements **Rayleigh fading** and **path loss models**.
  - Supports **noise power computation** based on system parameters.
- **Multiplicity and Type Estimation**:
  - Computes message **multiplicity vectors** for each zone and globally.
  - Normalizes estimates to obtain empirical **type distributions**.

### Main Functionalities:
- **Encoding & Transmission**:
  - `generate_codebook()`: Creates complex-valued normalized codebooks.
  - `transmit()`: Simulates the transmission and reception process.
- **Decoding & Estimation**:
  - `decoder()`: Calls either **centralized** or **distributed** decoders.
  - `compute_multiplicity_statistics()`: Computes detection statistics.
- **Visualization**:
  - `visualize_multiplicity_vector()`: Plots detected message multiplicities.

### Usage:
- Used as the **main simulation environment** for TUMA applications in **MTL** and **FL**.
- Calls **sensing (`sensing.py`)**, **decoding (`centralized_decoder.py`, `distributed_decoder.py`)**, and **prior computation (`prior.py`)**.
"""

import os
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore

from centralized_decoder import centralized_decoder
from distributed_decoder import distributed_decoder
from amp_da import AMP_DA
from prior import compute_prior

class TUMAEnvironment:
    """
    TUMA Environment: Handles communication, encoding, and decoding.
    - Manages multiplicity computation.
    - Simulates transmissions.
    - Calls centralized or distributed decoders.
    """
    def __init__(self, num_targets, num_sensors, active_sensors, topology, blocklength, codebook_size, num_antennas, M,
                 path_loss_exp, ref_distance, SNR_rx_dB, P, nAMPiter, N_s, N_MC, N_MC_smaller=1, perfect_comm=False, Kmax=None, 
                 decoder_type="centralized",
                 perfect_CSI=True, imperfection_model="phase", sigma_noise_e=1, phase_max=np.pi/6, keep_SNRrx_fixed=True):
        """
        Initializes the TUMA environment with given system parameters.
        
        Parameters:
        - num_targets: Number of targets in the system.
        - num_sensors: Number of sensors in the system.
        - active_sensors: List of Sensor objects that are actively transmitting.
        - topology: NetworkTopology object containing AP and zone information.
        - blocklength: Length of the transmission block.
        - codebook_size: Number of unique codewords per zone.
        - num_antennas: Number of antennas per AP.
        - M: Total number of possible messages.
        - path_loss_exp: Path loss exponent (rho) used in the fading model.
        - ref_distance: Reference distance for path loss calculation.
        - SNR_rx_dB: Received SNR at the APs in dB.
        - P: Transmission power of each sensor.
        - nAMPiter: Number of AMP iterations used in decoding.
        - N_s: Number of sensing blocklengths.
        - N_MC: Number of Monte Carlo simulations.
        - N_MC_smaller: Reduced number of Monte Carlo simulations for performance speedup.
        - perfect_comm: Boolean flag for using perfect communication (bypassing the decoding step).
        """

        # Store initialization parameters
        self.num_targets = num_targets
        self.num_sensors = num_sensors
        self.active_sensors = active_sensors
        self.topology = topology
        self.blocklength = blocklength
        self.N_s = N_s
        self.codebook_size = codebook_size
        self.num_aps = topology.B # Number of APs
        self.num_antennas = num_antennas
        self.M = M
        self.J = int(np.log2(M)) # Message length in bits
        self.F = self.num_aps * num_antennas  # Total antennas in network
        self.path_loss_exp = path_loss_exp
        self.ref_distance = ref_distance
        self.P = P
        self.nP = self.blocklength * self.P
        self.Y = None
        self.nAMPiter = nAMPiter
        self.N_MC = N_MC
        self.N_MC_smaller = N_MC_smaller
        self.perfect_comm = perfect_comm
        self.Kmax = num_targets if Kmax is None else Kmax
        self.decoder_type = decoder_type
        self.perfect_CSI = perfect_CSI
        self.keep_SNRrx_fixed = keep_SNRrx_fixed
        self.imperfection_model = imperfection_model
        self.sigma_noise_e = sigma_noise_e
        self.phase_max = phase_max

        # Compute noise power
        self.sigma_w = self.compute_noise_power(SNR_rx_dB)

        # Assign zones to sensors
        self.assign_zones()

        # Assign blocklength and codebooks to zones
        self.setup_zone_codebooks()

        # Compute multiplicity vectors
        self.multiplicity_per_zone = self.compute_multiplicity_per_zone()
        self.global_multiplicity = self.compute_global_multiplicity()
        self.Ka_true = np.sum(self.global_multiplicity)
        self.true_type = self.compute_global_type()

        # Store AP positions
        self.ap_positions = np.array([ap.position for ap in self.topology.aps])

        # Generate transmission matrix X
        self.X = self.generate_X()

    def compute_noise_power(self, SNR_rx_dB):
        """ Computes the noise power based on the received SNR and path loss model. """
        SNR_rx = 10**(SNR_rx_dB / 10)
        nus = np.array([ap.position for ap in self.topology.aps])
        min_dist = np.abs(np.array([(1+1j)*0.0]) - nus).min()
        SNR_tx = SNR_rx * (1 + (min_dist / self.ref_distance) ** self.path_loss_exp)
        return np.sqrt(self.P / SNR_tx)
    
    def assign_zones(self):
        """ Assigns each active sensor to a zone based on its position. """
        for sensor in self.active_sensors:
            zone_id = self.topology.get_zone_by_position(sensor.position)
            sensor.assigned_zone = self.topology.zones[zone_id]  # Store full zone object, not just ID

    def generate_codebook(self, blocklength, codebook_size):
        """ Generates a complex-valued normalized codebook matrix C. """
        C = np.random.randn(blocklength, codebook_size) + 1j * np.random.randn(blocklength, codebook_size)
        return C / np.linalg.norm(C, axis=0)

    def setup_zone_codebooks(self):
        """ Assigns blocklength and generates codebooks for each zone. """
        for zone in self.topology.zones:
            zone.blocklength = self.blocklength
            zone.codebook_size = self.codebook_size
            zone.C = self.generate_codebook(self.blocklength, self.codebook_size)

    def compute_multiplicity_per_zone(self):
        """ Computes the multiplicity vector for each zone. """
        multiplicity_per_zone = {}  
        for zone in self.topology.zones:
            Mu = zone.codebook_size  
            ku = np.zeros(Mu, dtype=int)  
            for sensor in self.active_sensors:
                if sensor.assigned_zone.id == zone.id:
                    for quantized_index in sensor.quantized_indices:
                        ku[quantized_index] += 1  
            multiplicity_per_zone[zone.id] = ku  
        return multiplicity_per_zone

    def compute_global_multiplicity(self):
        """ Computes the global multiplicity vector by summing local multiplicities. """
        global_k = np.zeros_like(next(iter(self.multiplicity_per_zone.values())), dtype=int)  
        for zone in self.topology.zones:
            global_k += self.multiplicity_per_zone[zone.id]  
        return global_k
    
    def compute_global_type(self):
        """ Computes the global type by normalizing the global multiplicity vector. """
        total_transmissions = np.sum(self.global_multiplicity)
        if total_transmissions == 0:
            return np.zeros_like(self.global_multiplicity, dtype=float)  # Avoid division by zero
        return self.global_multiplicity / total_transmissions

    def compute_path_loss(self, sensor_position):
        """ Computes path loss based on sensor position and AP locations. """
        distances = np.abs(sensor_position - self.ap_positions.reshape(-1, 1))  
        return 1 / (1 + (distances / self.ref_distance) ** self.path_loss_exp) # Path loss model

    def generate_X(self):
        """ Generates the transmission matrix X based on sensor positions and fading. """
        X = []  # List to store X_u matrices for each zone
        for zone in self.topology.zones:
            Mu = zone.codebook_size
            Xu = np.zeros((Mu, self.F), dtype=complex)  
            ku = self.multiplicity_per_zone[zone.id]  
            for m, ku_m in enumerate(ku):
                if ku_m > 0:
                    xu_m = np.zeros(self.F, dtype=complex)
                    # Get sensors that transmit this message
                    transmitting_sensors = [
                        sensor for sensor in self.active_sensors if sensor.assigned_zone.id == zone.id and m in sensor.quantized_indices
                    ]
                    for sensor in transmitting_sensors:
                        pos = sensor.position  
                        # Generate Rayleigh fading channel
                        h = (np.random.randn(self.F) + 1j * np.random.randn(self.F)) / np.sqrt(2)
                        # Apply path loss
                        h *= np.sqrt((self.compute_path_loss(pos) * np.ones((1, self.num_antennas))).reshape(-1))
                        # Channel inversion if using AMP-DA
                        if self.decoder_type == "AMP-DA":
                            if self.perfect_CSI:
                                h_e = h[0]  # Use exact channel vector
                            else:
                                if self.imperfection_model == "phase":
                                    phase = np.random.uniform(-self.phase_max, self.phase_max)
                                    h_e = h[0] * np.exp(1j * phase)
                                elif self.imperfection_model == "awgn":
                                    noise_e = (np.random.randn() + 1j * np.random.randn()) * (self.sigma_noise_e / np.sqrt(2))
                                    h_e = h[0] + noise_e
                                else:
                                    raise ValueError(f"Unknown imperfection_model: {self.imperfection_model}. Valid options: 'awgn', 'phase'.")
                            if self.keep_SNRrx_fixed:
                                norm_factor = np.abs(h_e)
                                h_e /= norm_factor # to keep SNR_rx same! (o.w., SNR_rx would boost!)
                            h /= h_e
                        # Accumulate transmission contributions
                        xu_m += h
                    Xu[m] = xu_m  
            X.append(Xu)
        return X

    def transmit(self):
        """ Simulates transmission and computes received signal Y. """
        W = (np.random.randn(self.blocklength, self.F) + 1j * np.random.randn(self.blocklength, self.F)) * np.sqrt(1 / 2) * self.sigma_w
        Y = W.copy()
        for zone in self.topology.zones:
            Y += np.sqrt(self.nP) * zone.Cx(self.X[zone.id])  # Compute \( Y = \sum C_u X_u + W \)
        self.Y = Y

    def get_codebook_functions(self):
        """ Returns global encoding (Cx) and decoding (Cy) functions for the decoder. """
        # Concatenate all zone codebooks to create a global codebook C
        C = np.hstack([zone.C for zone in self.topology.zones])  # Shape: (n, U*M)
        self.C = C
        
        def Cx(x, u=0):
            """ Encoding function: Projects x using the codebook of zone u. """
            start_idx = u * self.codebook_size
            end_idx = (u + 1) * self.codebook_size
            return C[:, start_idx:end_idx] @ x

        def Cy(y, u=0):
            """ Decoding function: Applies conjugate transpose projection for zone u. """
            start_idx = u * self.codebook_size
            end_idx = (u + 1) * self.codebook_size
            return (C[:, start_idx:end_idx].conj().T) @ y

        return Cx, Cy

    def load_or_compute_priors(self):
        """ Loads prior data if available, otherwise computes and saves it. """
        if os.getcwd().split(sep="/")[-1]=="experiments":
            prior_filename = f"../prior_data/priors_T{self.num_targets}_K{self.num_sensors}_J{self.J}_Ns{self.N_s}.npz"
        else:
            prior_filename = f"./prior_data/priors_T{self.num_targets}_K{self.num_sensors}_J{self.J}_Ns{self.N_s}.npz"
        if os.path.exists(prior_filename):
            print(f"Loading prior data from {prior_filename} ...")
            priors = np.load(prior_filename)["priors"]
        else:
            print(f"Computing prior data and saving to {prior_filename} ...")
            priors = compute_prior(self.num_sensors, self.num_targets, self.topology.U, self.N_s, self.M, zone_side=self.topology.side_length, include_Pclosest=False)
            np.savez(prior_filename, priors=priors)
        # Normalize the truncated priors
        for u in range(self.topology.U):
            for mu in range(self.M):
                sum_prob = np.sum(priors[u, mu, :self.Kmax+1])
                if sum_prob > 0:
                    priors[u, mu, :self.Kmax+1] /= sum_prob  # Normalize probabilities
                else:
                    priors[u, mu, :self.Kmax+1] = 1.0 / (self.Kmax + 1)  # Assign uniform distribution if all values are zero
        # Extract priors for each zone and compute log priors
        self.priors = [priors[u, :, :self.Kmax+1] for u in range(self.topology.U)]
        self.log_priors = [np.log(priors_u) for priors_u in self.priors]

    def generate_cov_matrices(self):
        """ Generates covariance matrices used in the decoding process. """
        zone_centers = np.array([zone.center for zone in self.topology.zones])
        _, all_covs = self.generate_all_covs(self.topology.side_length, self.ap_positions, zone_centers, self.num_antennas, self.F, self.path_loss_exp, self.ref_distance, Kmax=self.Kmax, Ns=self.N_MC)
        _, all_covs_smaller = self.generate_all_covs(self.topology.side_length, self.ap_positions, zone_centers, self.num_antennas, self.F, self.path_loss_exp, self.ref_distance, Kmax=self.Kmax, Ns=self.N_MC_smaller)
        self.all_covs = all_covs
        self.all_covs_smaller = all_covs_smaller

    def decoder(self, withOnsager=False, plot_perf=False):
        """ Decodes the received signal Y using either centralized or distributed decoder. """
        if self.Y is None:
            raise ValueError("There is no transmitted signal!")
        if self.perfect_comm:
            # i) perfect communication:
            estimated_global_multiplicity = self.global_multiplicity.copy()
        else:    
            # ii) noisy communication:
            Cx, Cy = self.get_codebook_functions()  # Get zone-specific encoding/decoding functions
            self.load_or_compute_priors() # Handle prior computation or loading
            self.generate_cov_matrices() # Generate cov matrices
            if self.decoder_type == "centralized":
                estimated_global_multiplicity, _ = centralized_decoder(Y=self.Y, M=self.M, U=self.topology.U, nAMPIter=self.nAMPiter, B=self.num_aps, A=self.num_antennas, Cx=Cx, Cy=Cy, nP=self.nP, priors=self.priors, log_priors=self.log_priors, all_Covs=self.all_covs, all_Covs_smaller=self.all_covs_smaller, withOnsager=withOnsager, 
                                            k_true=np.hstack(list(self.multiplicity_per_zone.values())), X_true=self.X, sigma_w=self.sigma_w, plot_perf=plot_perf)
            elif self.decoder_type == "distributed":
                estimated_global_multiplicity, _ = distributed_decoder(Y=self.Y, M=self.M, U=self.topology.U, nAMPIter=self.nAMPiter, B=self.num_aps, A=self.num_antennas, Cx=Cx, Cy=Cy, nP=self.nP, priors=self.priors, log_priors=self.log_priors, all_Covs=self.all_covs, all_Covs_smaller=self.all_covs_smaller, withOnsager=withOnsager, 
                                            k_true=np.hstack(list(self.multiplicity_per_zone.values())), X_true=self.X, sigma_w=self.sigma_w, plot_perf=plot_perf)
            elif self.decoder_type == "AMP-DA":
                estimated_global_multiplicity = AMP_DA(np.expand_dims(self.Y,axis=1), self.C, self.Ka_true, maxIte=50) 
            else:
                raise ValueError("The decoder type must be either 'centralized', 'distributed' or 'AMP-DA'!")
        self.estimated_global_multiplicity = estimated_global_multiplicity
        self.estimated_type = estimated_global_multiplicity/(estimated_global_multiplicity.sum()) if estimated_global_multiplicity.sum()!=0 else np.zeros(self.M)

    def compute_multiplicity_statistics(self):
        """ Computes detailed statistics for each zone and globally. """
        zone_stats = {}
        total_unique_messages = 0
        total_transmissions = 0
        for zone in self.topology.zones:
            ku = self.multiplicity_per_zone[zone.id]  # Multiplicity vector for this zone
            unique_messages = np.count_nonzero(ku)  # Number of unique messages
            total_sensors = np.sum(ku)  # Total number of transmissions (active sensors in zone)
            zone_stats[zone.id] = {
                "unique_messages": unique_messages,
                "total_sensors": total_sensors
            }
            total_unique_messages += unique_messages
            total_transmissions += total_sensors
        # Global statistics
        global_stats = {
            "total_unique_messages": total_unique_messages,
            "total_transmissions": total_transmissions
        }
        return zone_stats, global_stats

    def visualize_multiplicity_vector(self):
        """ Visualizes the global multiplicity vector across all messages. """
        plt.figure(figsize=(10, 5))
        plt.stem(range(len(self.global_multiplicity)), self.global_multiplicity)
        plt.xlabel("Message Index")
        plt.ylabel("Multiplicity")
        plt.title("Global Multiplicity Vector (Number of Times Each Message is Transmitted)")
        plt.grid(True)
        plt.show()

    def visualize_multiplicity_vector_per_zone(self):
        """ Visualizes the global multiplicity vector across all messages. """
        for zone_id in range(self.topology.U):
            plt.figure(figsize=(10, 5))
            plt.stem(range(len(self.multiplicity_per_zone[zone_id])), self.multiplicity_per_zone[zone_id])
            plt.xlabel("Message Index")
            plt.ylabel("Multiplicity")
            plt.title(f"Multiplicity Vector for Zone {zone_id+1}")
            plt.grid(True)
            plt.show()

    def visualize_type_estimation(self):
        """ Visualizes the true and estimated type together. """
        plt.figure(figsize=(10, 5))
        plt.stem(range(len(self.true_type)), self.true_type, "k", label="true")
        plt.stem(range(len(self.estimated_type)), self.estimated_type, "r--", label="est")
        plt.legend()
        plt.xlabel("Message Index")
        plt.ylabel("True and Estimated Type")
        plt.grid(True)
        plt.show()

    def gamma(self, q, v=0.0 + 1j * 0.0, rho=3.67, d0=0.01357):
        """ Compute large-scale fading coefficients based on distance. """
        return 1 / (1 + (np.abs(q - v) / d0) ** rho)

    def generate_cov_matrix(self, pos, nus, A, rho=3.67, d0=0.01357):
        """ Generate covariance matrix based on user positions. """
        pos_shape = list(pos.shape)
        return (self.gamma(np.expand_dims(pos,axis=(-1,-2)), nus.reshape(-1,1), rho=rho, d0=d0)*np.ones((1,A))).reshape(pos_shape + [-1])

    def generate_uniform_grid(self, side, num_points, zone_centers, margin=0):
        """ Generate a uniform grid of points for all zones. """
        # Calculate approximate number of points per axis
        num_per_axis = int(np.sqrt(num_points))

        # Generate grid points
        x = np.linspace(-side/2 + margin, side/2 - margin, num_per_axis)
        y = np.linspace(-side/2 + margin, side/2 - margin, num_per_axis)
        xx, yy = np.meshgrid(x, y)
        base_qs = xx.ravel() + 1j * yy.ravel()

        # Generate grid points for each zone
        Qus = np.zeros([len(zone_centers)] + list(base_qs.shape), dtype=complex)
        for u, zone_center in enumerate(zone_centers):
            Qus[u] = zone_center + base_qs

        return Qus

    def generate_all_covs(self, side, nus, zone_centers, A, F, rho, d0, Kmax, Ns, num_samples=2000):
        """ Generate covariance matrices for all zones using sampling-based approximation. """
        U = len(zone_centers)
        Qus = self.generate_uniform_grid(side, num_samples, zone_centers)
        positions_for_ks = [np.array([np.vstack([np.random.choice(Qus[u], size=(k)) for _ in range(Ns)]) for u in range(U)]) for k in range(1,Kmax+1)]
        all_Covs = np.array([self.generate_cov_matrix(positions_for_ks[k], nus, A, rho=rho, d0=d0).sum(axis=-2) for k in range(Kmax)])
        all_Covs = np.vstack((np.zeros((1, U, Ns, F)), all_Covs))
        all_Covs = np.vstack([np.expand_dims(all_Covs[:,u,:,:],axis=0) for u in range(U)])
        return positions_for_ks, all_Covs
