# Type-Based Unsourced Multiple Access Over Fading Channels in Distributed MIMO  With Application to Multi-Target Localization

This repository contains the research code for our journal manuscript on **type-based unsourced multiple access (TUMA)** over fading channels in distributed MIMO (D-MIMO) with application to **multi-target localization (MTL)**. 


The framework integrates sensing, quantization, and TUMA-based uplink communication to analyze sensing–communication tradeoffs under centralized and distributed decoding. The implementation is fully simulation-based and supports Monte Carlo (MC) experiments to evaluate detection, communication, and localization performance.


## Key Features
- **TUMA over Fading Channels with D-MIMO:** Multisource AMP decoding under Rayleigh fading without requiring CSI at users or receiver, with explicit modeling and recovery of message collisions via a type-based formulation. 

- **Centralized and Distributed Decoders:** Implementation of both centralized and distributed multisource AMP decoders. Optional comparison with AMP-DA decoder

- **Multi-Target Localization Pipeline:** End-to-end simulation from target detection to quantized position reporting and type decoding.

- **Performance Metrics:**
    - **Total variation distance:** Measures type estimation error (communication performance).
    - **Empirical misdetection probability:** Measures the probability of missed active messages (sensing performance).
    - **Wasserstein distance**: Captures localization error due to quantization and communication effects.
    - **GOSPA-like cost:** Joint metric accounting for sensing errors, quantization error, and communication errors.



## Environment
This project was developed and tested with Python 3.11.8. To set up the environment, it’s recommended to create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

Then, install the required libraries:
- **Numpy:** For matrix and vector operations. Install with `pip install numpy`.
- **Scipy:** For probability distributions and pmf calculations. Install with `pip install scipy`.
- **Matplotlib:**  For data visualization. Install with `pip install matplotlib`.

## Project Structure 

The project structure is as follows:
```bash
./
├── README.md                       # Project overview and usage instructions
│
├── TUMA_MTL_demo.ipynb             # Interactive demo notebook (visuals + runnable cells)
│
├── run_tuma_mtl.py                 # CLI/entry script to run the simulation end-to-end
│
├── tuma_mtl/
│   ├── __init__.py
│   ├── sensing.py                  # Simulate sensor-target interactions & detection logic
│   ├── topology.py                 # Zone/AP placement and network topology utilities
│   ├── quant.py                    # Quantization utilities
│   ├── tuma.py                     # High-level TUMA environment and helpers
│   ├── devices.py                  # Device classes: Target, Sensor, AccessPoint
│   ├── prior.py                    # Prior computation / loading (e.g., prior_data)
│   ├── bayesian_denoiser.py        # Bayesian denoiser implementation with Onsager term
│   ├── amp_da.py                   # AMP-DA decoder (comparison)
│   ├── centralized_decoder.py      # Centralized multisource AMP decoder for TUMA
│   ├── distributed_decoder.py      # Distributed multisource AMP decoder for TUMA
│   ├── metrics.py                  # Performance metrics (TV, Wasserstein, GOSPA-like cost)
│   └── tuma_mtl_simulation.py      # High-level experiment runner / orchestrator
│
└── logs/                           # Runtime logs and experiment traces
    └── .gitkeep
└── prior_data/                     # Generated or precomputed priors 
    └── .gitkeep
```

## Usage

You can run the project in two modes of execution:

1. **Full MC Experiments (`run_tuma_mtl.py`):** Run complete MC simulations: 
    ```
    python run_tuma_mtl.py
    ```
    This will
    - Run the sensing → quantization → communication → decoding pipeline.
    - Precompute priors if not already available
    - Log results per MC run
    - Save metrics as CSV files under `./logs`.

    To customize system parameters, modify the parameters in `run_tuma_mtl.py` as desired.


2. **Interactive Single-Run Demo (`TUMA_MTL_demo.py`):** For step-by-step inspection of a single MC realization, open the notebook `TUMA_MTL_demo.py`. This notebook: 
	- Walks through topology generation
	- Simulates sensing
	- Applies quantization
	- Runs TUMA decoding
	- Computes performance metrics
	- Visualizes intermediate results

    This mode is recommended for understanding the full pipeline, debugging, and small-scale experimentation. 

    To customize the system parameters, change the system parameters in `Setting Parameters` section in `TUMA_MTL_demo.ipynb`.



## Citation

This work is based on our journal manuscript, which has not been published yet. Citation details will be added once the paper is published.

 
## References

- **TUMA**:
  K.-H. Ngo, D. P. Krishnan, K. Okumus, G. Durisi, and E. G. Ström, "Type-Based Unsourced Multiple Access", *IEEE Workshop on Signal Processing Advances in Wireless Communications (SPAWC)*, 2024.  [https://ieeexplore.ieee.org/document/10694658](https://ieeexplore.ieee.org/document/10694658)

- **TUMA over fading channels with CF massive MIMO**:
  K. Okumus, K.-H. Ngo, G. Durisi, and E. G. Ström, "Type-Based Unsourced Multiple Access Over Fading Channels with Cell-Free Massive MIMO", *IEEE International Symposium on Information Theory (ISIT)*, 2025.  [https://ieeexplore.ieee.org/document/11195493](https://ieeexplore.ieee.org/document/11195493)

- **Multisource AMP Algorithm**:
  B. Cakmak, E. Gkiouzepi, M. Opper, and G. Caire, "Joint Message Detection and Channel Estimation for Unsourced Random Access in Cell-Free User-Centric Wireless Networks," *IEEE Transactions on Information Theory*, 2025. [https://ieeexplore.ieee.org/document/10884602](https://ieeexplore.ieee.org/document/10884602)

- **MD-AirComp's AMP-DA Algorithm**:
  L. Qiao, Z. Gao, M. B. Mashadi, and D. Gunduz "Digital Over-the-Air Aggregation for Federated Edge Learning," *IEEE Journal on Selected Areas in Communications*, 2024. [https://ieeexplore.ieee.org/document/10648926](https://ieeexplore.ieee.org/document/10648926)

- **Code References**:
  + TUMA scheme codes from [okumuskaan/tuma_fading_cf](https://github.com/okumuskaan/tuma_fading_cf).
  + AMP-DA codes based on [liqiao19/MD-AirComp](https://github.com/liqiao19/MD-AirComp).
