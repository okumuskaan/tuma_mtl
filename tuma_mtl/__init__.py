"""
TUMA-MTL: Type-Based Unsourced Multiple Access for Multi-Target Localization

Core modules for sensing, quantization, TUMA communication, and decoding.
"""

from .sensing import SensingEnvironment
from .topology import NetworkTopology
from .quant import QuantizationEnvironment
from .tuma import TUMAEnvironment
from .metrics import tv_distance, get_true_positions_and_type, wasserstein_distance, compute_tuma_mtl_performance_metric
from .tuma_mtl_simulation import TUMA_MTL_Simulation