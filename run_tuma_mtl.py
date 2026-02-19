"""
run_tuma_mtl.py

Author: Kaan Okumus
Date: March 2025

This script runs **Multi-Target Localization (MTL) with TUMA communication** by initializing 
and executing the `TUMA_MTL_Simulation`. It integrates:
- **Sensing** (target detection using probabilistic models),
- **Quantization** (position encoding),
- **TUMA decoding** (centralized or distributed).

### Parameters:
- Number of sensors & targets.
- Sensing blocklength (`N_s`), Communication blocklength (`N_c`).
- Quantization resolution (`J` bits, `M=2^J` codewords).
- Power constraints (`P_s` for sensing, `P_c` for communication).
- Noise, environment, and path loss settings.
- Decoding parameters (AMP iterations, Monte Carlo runs).
"""

from tuma_mtl_simulation import TUMA_MTL_Simulation

# ------------------------- Simulation Parameters -------------------------

# Number of sensors and targets
num_sensors = 200 
num_targets = 50  

# Sensing and Communication Blocklengths
N_s = 1000 # sensing blocklength
N_c = 2000 - N_s  # communication blocklength (total blocklength 2000)

# Quantization parameters
J = 10 # number of quantization bits 
M = 2**J # number of possible codewords

# Power constraints
P_s = 1 / 1000 # sensing power
P_c = 1 / 1000 # communication power

# SNR at the receiver
SNR_rx_dB = 10.0 

# Experiment type
perfect_comm = False  # if True, assumes perfect communication (no decoding needed)
decoder_type = "distributed"  # choose between "centralized" and "distributed"

# Environment settings
zone_side = 0.1  # side length of each zone
area_side = 3 * zone_side  # total sensing area (3x3 grid of zones)

# Sensing noise and parameters
sigma_n = 10**(-5) # thermal noise for sensing
S = 100 # radar cross-section area
fc = 2.8e9 # carrier frequency
gamma = 36.841361487904734 # detection threshold (corresponds to p_fa = 1e-8)

# Sensor detection settings
max_detections_per_sensor = 1  # maximum targets a sensor can detect
perfect_measurement = True  # if True, no measurement noise
sigma_noise = None  # measurement noise (ignored if perfect_measurement=True)

# Communication settings
A = 4 # number of antennas per AP
rho = 3.67 # path loss exponent
d0 = 0.01357 # reference distance for path loss calculation

# Algorithm parameters
nAMPiter = 10 # number of AMP iterations
N_MC = 500 # number of MC samples for covariance approximations

# Number of Monte Carlo runs
nMCs = 100

# ------------------------- Run the Simulation -------------------------

# Create simulation instance
tuma_mtl_sim = TUMA_MTL_Simulation(
    num_sensors=num_sensors, num_targets=num_targets, 
    area_side=area_side, zone_side=zone_side, 
    N_s=N_s, P_s=P_s, 
    sigma_n=sigma_n, S=S, fc=fc, gamma=gamma, 
    max_detections_per_sensor=max_detections_per_sensor, 
    perfect_measurement=perfect_measurement, sigma_noise=sigma_noise, 
    M=M, A=A, rho=rho, d0=d0, SNR_rx_dB=SNR_rx_dB, 
    N_c=N_c, P_c=P_c, 
    nMCs=nMCs, perfect_comm=perfect_comm, 
    nAMPIter=nAMPiter, N_MC=N_MC, decoder_type=decoder_type
)

# Run the experiment
tuma_mtl_sim.run()
print("✅ Simulation done!")
