"""
devices.py

Author: Kaan Okumus
Date: March 2025

This module defines the **sensor and target devices** used in the **Sensing Environment** for multi-target localization (MTL). 
It models how sensors operate, detect targets, and interact with the environment.

### Key Components:
- **Sensor Class**: Represents a sensing device that detects targets within a given range.
  - Each sensor has a position, noise characteristics, and detection probability function.
  - Supports probabilistic detection based on the Marcum Q-function.
  - Can be configured for perfect or noisy measurements.
- **Target Class**: Represents a target that sensors attempt to detect.
  - Each target has a unique ID and a fixed position in the environment.

### Main Functionalities:
- **Sensors**
  - `detect_targets()`: Determines which targets are detected based on distance and detection probability.
  - `quantize_position()`: Converts a detected target's position into a discrete quantization index.
- **Targets**
  - Targets remain passive and are only detected by sensors.

### Usage:
- Used within **Sensing Environment** (`sensing.py`) to simulate real-world target detection.
- Provides **ground truth positions** and **detection statistics** for MTL applications.
"""

import numpy as np # type: ignore

class Target:
    """ Represents a target in the environment. """
    def __init__(self, target_id, position):
        self.id = target_id # Unique ID
        self.position = position  # Complex coordinate (x + 1j*y)
        self.is_active = False # Becomes active if detected by any sensor

class Sensor:
    """ Represents a sensor that detects targets probabilistically. """
    def __init__(self, sensor_id, position, sigma_noise, P_d_func, sensing_radius, perfect_measurement=False, max_detections_per_sensor=None):
        self.id = sensor_id # Unique ID
        self.position = position  # Complex coordinate (x + 1j*y)
        self.sigma_noise = sigma_noise  # Measurement noise
        self.perfect_measurement = perfect_measurement # Toggle for perfect/noisy measurements
        self.max_detections_per_sensor = max_detections_per_sensor # Maximum number of targets a sensor can detect
        self.P_d_func = P_d_func  # Detection probability function
        self.sensing_radius = sensing_radius/1000 # Computed sensing radius based on detection probability function that will be used in visualization
        self.detected_targets = {}  # Stores detected targets and noisy measurements
        self.num_detected = 0 # Number of detected targets
        self.is_active = False # Active if it detects at least one target
        self.quantized_indices = []  
        self.quantized_data = []  

    def detect_targets(self, targets):
        """ Detects targets within its sensing radius probabilistically. """
        self.detected_targets.clear()
        detected_candidates = []  # Stores (target, distance, measurement)

        for target in targets:
            distance = np.abs(self.position - target.position)

            #if distance < self.sensing_radius:
            # Compute detection probability using the provided function
            P_detect = self.P_d_func(distance*1000)
            
            # Probabilistic detection
            if np.random.rand() < P_detect:
                # Apply measurement model (perfect or noisy)
                if self.perfect_measurement:
                    measurement = target.position  # No noise
                else:
                    measurement = target.position + np.random.normal(0, self.sigma_noise) + 1j * np.random.normal(0, self.sigma_noise)
                
                # Store candidate detection
                detected_candidates.append((target, distance, measurement))

        # If max detections per sensor is defined, select the closest ones
        if self.max_detections_per_sensor is not None and len(detected_candidates) > self.max_detections_per_sensor:
            #detected_candidates = [detected_candidates[np.random.randint(len(detected_candidates))]]
            detected_candidates.sort(key=lambda x: x[1])  # Sort by distance
            detected_candidates = detected_candidates[:self.max_detections_per_sensor]  # Keep closest
        
        # Store the final detections
        for target, _, measurement in detected_candidates:
            self.detected_targets[target.id] = measurement  
            target.is_active = True  # Mark target as active

        # Update detection count and activity status
        self.num_detected = len(self.detected_targets)
        self.is_active = self.num_detected > 0

class AccessPoint:
    """ Represents an Access Point (AP) positioned at a zone boundary. """
    def __init__(self, ap_id, position):
        self.id = ap_id  # Unique ID
        self.position = position  # Complex coordinate (x + 1j*y)
