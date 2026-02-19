"""
tuma_mtl_simulation.py

Author: Kaan Okumus  
Date: March 2025  

This module **simulates Multi-Target Localization (MTL) with TUMA-based communication**.  
It integrates sensing, quantization, and decoding to evaluate the full pipeline from target detection  
to message transmission and decoding in a **massive random access scenario**.

### Key Components:
- **`TUMA_MTL_Simulation`**:  
  - Orchestrates the full simulation, including sensing, quantization, and communication.
  - Runs Monte Carlo (MC) experiments to evaluate performance across multiple trials.

### Simulation Workflow:
1. **Sensing Stage** (`SensingEnvironment`):
   - Sensors detect targets based on **probabilistic detection functions**.
   - Detected target positions are stored.

2. **Quantization Stage** (`QuantizationEnvironment`):
   - Detected target positions are **quantized** into discrete values.

3. **TUMA Communication Stage** (`TUMAEnvironment`):
   - Quantized positions are mapped to messages.
   - Messages are **transmitted unsourced** to Access Points (APs).
   - The received signal is processed for decoding.

4. **Decoding Stage** (`centralized_decoder.py` / `distributed_decoder.py`):
   - Messages are decoded using **Bayesian denoising and AMP algorithms**.
   - Estimated target positions and their multiplicities are reconstructed.

5. **Performance Evaluation** (`metrics.py`):
   - Computes:
     - **TV distance**: Measures communication errors.
     - **Empirical misdetection probability**: Evaluates sensing performance.
     - **Wasserstein distance**: Assesses overall estimation error.
   - Logs results for later analysis.

### Main Functionalities:
- **`run_single_simulation(idxMC)`**:  
  - Runs a single **Monte Carlo simulation**.
  - Initializes the **sensing, quantization, communication, and decoding** steps.
  - Computes performance metrics and logs results.

- **`log_performance(idxMC, tv_dist, p_md, ws_distance)`**:  
  - Saves results to structured **CSV log files** for reproducibility.

- **`run()`**:  
  - Runs the full **Monte Carlo simulation** loop.

- **`get_results()`**:  
  - Returns stored performance metrics.

- **`run_to_get_pmd()`**:  
  - Computes the **empirical probability of missed detection**.

### Usage:
This module serves as the **main simulation script** for evaluating **joint sensing and communication**  
using MTL and TUMA. It integrates multiple components (`sensing.py`, `quant.py`, `tuma.py`, etc.)  
to provide a **full system simulation**.
"""

import os
import datetime
import numpy as np # type: ignore

from sensing import SensingEnvironment
from topology import NetworkTopology
from quant import QuantizationEnvironment
from tuma import TUMAEnvironment
from metrics import *

class TUMA_MTL_Simulation:
    """
    Simulates Multi-Target Localization (MTL) with TUMA communication,
    handling sensing, quantization, and decoding.
    """

    def __init__(self, num_sensors, num_targets, area_side, zone_side, sigma_noise,
                 N_s, P_s, sigma_n, S, fc, gamma,
                 perfect_measurement, max_detections_per_sensor, N_c, M, A, rho, d0, 
                 SNR_rx_dB, P_c, nAMPIter, nMCs, 
                 N_MC, N_MC_smaller=1, perfect_comm=False, decoder_type="centralized", withOnsager=False, Kmax=25, plot_perf=False, boxplot_flag=False,
                 perfect_CSI=True, imperfection_model="phase", sigma_noise_e=1, phase_max=np.pi/6, keep_SNRrx_fixed=True):
        """
        Initializes simulation parameters.
        
        Parameters:
        - num_sensors: Number of sensors.
        - num_targets: Number of targets.
        - area_side: Side length of the area (square region).
        - zone_side : Side length of the zone (square region).
        - sigma_noise: Standard deviation of measurement noise.
        - N_s: Sensing blocklength.
        - P_s: Sensing power.
        - sigma_n: Noise power.
        - S: Target cross-section area.
        - fc: Carrier frequency.
        - gamma: Detection threshold parameter.
        - perfect_measurement: If True, sensors measure without noise.
        - max_detections_per_sensor: Maximum detections per sensor.
        - N_c: Communication blocklength.
        - M: Number of possible messages.
        - A: Number of antennas per AP.
        - rho: Path loss exponent used in the fading model.
        - Reference distance for path loss calculation.
        - SNR_rx_dB: Received SNR at the APs in dB.
        - P_c: Communication power
        - nAMPIter: Number of AMP iterations used in decoding.
        - nMCs: Number of MC experiments for averaging performances.
        - N_MC: Number of Monte Carlo simulations.
        - N_MC_smaller: Reduced number of Monte Carlo simulations for performance speedup.
        - perfect_comm: Boolean flag for using perfect communication (bypassing the decoding step).
        - decoder_type: String for determining which decoder to be used. Either 'centralized' or 'distributed'.
        - withOnsager: Boolean variable for including Onsager term.
        """
        self.num_sensors = num_sensors
        self.num_targets = num_targets
        self.area_side = area_side
        self.zone_side = zone_side
        self.N_s = N_s
        self.P_s = P_s
        self.sigma_n = sigma_n
        self.S = S
        self.fc = fc
        self.gamma = gamma
        self.perfect_measurement = perfect_measurement
        self.sigma_noise = sigma_noise
        self.max_detections_per_sensor = max_detections_per_sensor
        self.N_c = N_c
        self.M = M
        self.J = int(np.log2(self.M))
        self.A = A
        self.rho = rho
        self.d0 = d0
        self.SNR_rx_dB = SNR_rx_dB
        self.P_c = P_c
        self.N_MC = N_MC
        self.N_MC_smaller = N_MC_smaller
        self.nAMPIter = nAMPIter
        self.nMCs = nMCs
        self.perfect_comm = perfect_comm
        self.decoder_type = decoder_type
        self.withOnsager = withOnsager
        self.Kmax = Kmax
        self.plot_perf = plot_perf
        self.boxplot_flag = boxplot_flag
        self.perfect_CSI = perfect_CSI
        self.imperfection_model = imperfection_model
        self.sigma_noise_e = sigma_noise_e
        self.phase_max = phase_max
        self.keep_SNRrx_fixed = keep_SNRrx_fixed

        # Store performance results
        self.tv_dists = np.zeros(nMCs)
        self.ws_dists = np.zeros(nMCs)
        self.empirical_p_mds = np.zeros(nMCs)

        # Store environments
        self.sensing_env = None
        self.topology = None
        self.quant_env = None
        self.TUMA_env = None

        # Initialize Network Topology
        self.topology = NetworkTopology(side_length=self.zone_side)

    def run_single_simulation(self, idxMC):
        """ Runs a single Monte Carlo simulation and records performance metrics. """
        print(f"Sim: {idxMC+1}/{self.nMCs}")

        # 1. Initialize Sensing Environment
        self.sensing_env = SensingEnvironment(
            self.num_sensors, self.num_targets, self.area_side, self.sigma_noise, 
            perfect_measurement=self.perfect_measurement, max_detections_per_sensor=self.max_detections_per_sensor,
            Ns=self.N_s, Ps=self.P_s, sigma_n=self.sigma_n, S=self.S, fc=self.fc, gamma=self.gamma
        )

        # 2. Run Sensing Simulation
        self.sensing_env.run_sensing()

        # 3. Initialize Network Topology
        self.topology = NetworkTopology(side_length=self.zone_side)

        # 4. Get Targets & Active Sensors
        targets, active_sensors = self.sensing_env.get_targets_and_active_sensors()

        # 5. Initialize Quantization Environment & Apply Quantization
        self.quant_env = QuantizationEnvironment(active_sensors, quantization_levels=self.M, area_side=self.area_side)
        self.quant_env.apply_quantization()

        # 6. Initialize TUMA Environment
        self.TUMA_env = TUMAEnvironment(
            self.num_targets, self.num_sensors,
            active_sensors, self.topology, blocklength=self.N_c, codebook_size=self.M,
            num_antennas=self.A, M=self.M, path_loss_exp=self.rho, ref_distance=self.d0,
            SNR_rx_dB=self.SNR_rx_dB, P=self.P_c, nAMPiter=self.nAMPIter, N_s=self.N_s,
            N_MC=self.N_MC, N_MC_smaller=self.N_MC_smaller, perfect_comm=self.perfect_comm, Kmax=self.Kmax,
            decoder_type=self.decoder_type,
            perfect_CSI=self.perfect_CSI, imperfection_model=self.imperfection_model,
            sigma_noise_e=self.sigma_noise_e, phase_max=self.phase_max, keep_SNRrx_fixed=self.keep_SNRrx_fixed
        )
        if self.boxplot_flag:
            self.mults = [
                list(self.TUMA_env.multiplicity_per_zone[u][np.nonzero(self.TUMA_env.multiplicity_per_zone[u])]) for u in range(self.topology.U)
            ]

        # 7. Transmit & Obtain Received Signal
        self.TUMA_env.transmit()

        # 8. TUMA Decoder
        self.TUMA_env.decoder(withOnsager=self.withOnsager, plot_perf=self.plot_perf)

        # 9. Compute Performance Metrics
        try:
            # i) tv distance for comm errors:
            tv_dist = tv_distance(self.TUMA_env.true_type, self.TUMA_env.estimated_type)

            # ii) empirical misdetection probability for sensing errors:
            true_positions = np.array([target.position for target in targets])
            detected_positions, _, true_target_type = get_true_positions_and_type(active_sensors)
            p_md = 1 - len(detected_positions)/len(true_positions)
            
            # iii) wasserstein distance for comm + quant errors:
            estimated_positions = self.quant_env.quantized_positions
            ws_distance = wasserstein_distance(
                detected_positions.reshape(-1, 1),
                estimated_positions.reshape(-1, 1),
                true_target_type.reshape(-1, 1),
                self.TUMA_env.estimated_type.reshape(-1, 1)
            )

            # Store results
            self.tv_dists[idxMC] = tv_dist
            self.empirical_p_mds[idxMC] = p_md
            self.ws_dists[idxMC] = ws_distance

            # Log performance values to a file
            self.log_performance(idxMC, tv_dist, p_md, ws_distance)

        except Exception as e:
            print(f"An error occurred: {e}")

    def log_performance(self, idxMC, tv_dist, p_md, ws_distance):
        """ Logs performance results to a structured CSV file. """
        # Determine the correct folder based on communication type
        if self.perfect_comm:
            log_subdir = "perfect_comm"
        elif self.decoder_type == "centralized":
            log_subdir = "centralized"
        elif self.decoder_type == "distributed":
            log_subdir = "distributed"
        elif self.decoder_type == "AMP-DA":
            log_subdir = "AMP-DA"
        else:
            raise ValueError(f"Unknown decoder type: {self.decoder_type}")
        # Create the full directory path
        if os.getcwd().split(sep="/")[-1]=="experiments":
            log_dir = os.path.join("../logs", log_subdir)
        else:
            log_dir = os.path.join("./logs", log_subdir)
        os.makedirs(log_dir, exist_ok=True)  # Ensure logs directory exists
        # Define log file name based on simulation parameters
        if self.decoder_type == "AMP-DA":
            if self.perfect_comm:
                log_filename = f"{log_dir}/results_Ns{self.N_s}_T{self.num_targets}_K{self.num_sensors}_J{self.J}_SNRrx{int(self.SNR_rx_dB)}.csv"
            else:
                if self.perfect_CSI:
                    phase_max = 0
                else:
                    phase_max = self.phase_max
                log_filename = f"{log_dir}/results_Ns{self.N_s}_Nc{self.N_c}_T{self.num_targets}_K{self.num_sensors}_J{self.J}_SNRrx{int(self.SNR_rx_dB)}_Phasemax{phase_max}.csv"
        else:
            if self.perfect_comm:
                log_filename = f"{log_dir}/results_Ns{self.N_s}_T{self.num_targets}_K{self.num_sensors}_J{self.J}_SNRrx{int(self.SNR_rx_dB)}.csv"        
            else:
                log_filename = f"{log_dir}/results_Ns{self.N_s}_Nc{self.N_c}_T{self.num_targets}_K{self.num_sensors}_J{self.J}_SNRrx{int(self.SNR_rx_dB)}.csv"
        # Check if file exists to write headers only once
        file_exists = os.path.exists(log_filename)
        with open(log_filename, "a") as f:
            if idxMC == 0 or not file_exists:
                f.write("timestamp,MC_idx,tv_dist,p_md,ws_distance\n")  # Write header if file is new or it's the first MC simulation
            # Get current timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Write data entry
            f.write(f"{timestamp},{idxMC},{tv_dist},{p_md},{ws_distance}\n")
        print(f"\tLogged results to {log_filename}")

    def run(self):
        """ Runs the full Monte Carlo simulation. """
        if self.boxplot_flag:
            self.all_mults = [[] for _ in range(self.topology.U)]
        for idxMC in range(self.nMCs):
            self.run_single_simulation(idxMC)
            if self.boxplot_flag:
                for u in range(self.topology.U):
                    self.all_mults[u] += self.mults[u]
        return self.tv_dists, self.empirical_p_mds, self.ws_dists

    def get_results(self):
        """ Returns the stored performance metrics. """
        return self.tv_dists, self.empirical_p_mds, self.ws_dists

    def run_to_get_pmd(self):
        n_missed_detections = np.zeros(self.nMCs)
        for idxMC in range(self.nMCs):
            print(f"\tSim: {idxMC+1}/{self.nMCs}")
            sensing_env = SensingEnvironment(
                self.num_sensors, self.num_targets, self.area_side, self.sigma_noise, 
                perfect_measurement=self.perfect_measurement, max_detections_per_sensor=self.max_detections_per_sensor,
                Ns=self.N_s, Ps=self.P_s, sigma_n=self.sigma_n, S=self.S, fc=self.fc, gamma=self.gamma
            )
            sensing_env.run_sensing()
            n_active_targets = np.sum([target.is_active for target in sensing_env.targets])
            print(f"\tn_active_targets={n_active_targets}")
            n_missed_detections[idxMC] = self.num_targets - n_active_targets
        return n_missed_detections.mean()/self.num_targets
