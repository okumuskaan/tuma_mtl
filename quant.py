"""
quant.py

Author: Kaan Okumus  
Date: March 2025  

This module defines the **Quantization Environment** for multi-target localization (MTL).  
It applies **2D uniform grid quantization** to detected target positions for active sensors.

### Key Components:
- **QuantizationEnvironment**: Manages the quantization process for detected targets.
  - Uses a **2D uniform grid** for quantizing detected target positions.
  - Computes **grid centroids** for precise mapping.
  - Applies **nearest centroid mapping** to quantize sensor measurements.
  - Provides tools for converting between quantization indices and positions.

### Main Functionalities:
- **`compute_grid_centroids()`**: Computes the centroids of quantization grid cells.
- **`apply_quantization()`**: Applies quantization to each active sensor’s detected targets.
- **`quantize(measurements)`**: Maps detected target positions to the closest quantization grid centroid.
- **`get_positions_from_indices(indices)`**: Retrieves the quantized positions corresponding to given indices.
- **Visualization Functions**:
  - `visualize_quantization()`: Plots the quantization grid along with original and quantized positions.

### Usage:
This module is used in **TUMA-based MTL applications**, where sensor-detected target positions  
need to be mapped to discrete indices for efficient communication and decoding.
"""

import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore

class QuantizationEnvironment:
    """ Handles quantization of detected target positions for active sensors using a 2D uniform grid. """
    def __init__(self, active_sensors, quantization_levels, area_side):
        """
        Parameters:
        - active_sensors: List of active Sensor objects.
        - quantization_levels: Number of quantization levels (must be 2^even number).
        - area_side: The side length of the total quantized area.
        """
        self.active_sensors = active_sensors
        self.quantization_levels = quantization_levels
        self.area_side = area_side
        self.sqrt_Q = int(np.sqrt(quantization_levels))  # Grid dimension: sqrt(Q) x sqrt(Q)
        self.cell_size = area_side / self.sqrt_Q  # Size of each grid cell
        self.quantization_centers = self.compute_grid_centroids()
        self.quantized_positions = self.quantization_centers.flatten()

    def compute_grid_centroids(self):
        """ Computes the centroids of each quantization grid cell. """
        step = self.cell_size
        x_positions = np.linspace(-self.area_side / 2 + step / 2, self.area_side / 2 - step / 2, self.sqrt_Q)
        y_positions = np.linspace(-self.area_side / 2 + step / 2, self.area_side / 2 - step / 2, self.sqrt_Q)
        grid_x, grid_y = np.meshgrid(x_positions, y_positions)
        return grid_x + 1j * grid_y  # Complex representation

    def apply_quantization(self):
        """ Applies 2D quantization to the detected target positions for each active sensor. """
        for sensor in self.active_sensors:
            sensor.quantized_indices, sensor.quantized_data = self.quantize(sensor.detected_targets.values())

    def quantize(self, measurements):
        """ Maps detected positions to the nearest quantization grid centroids. """
        if not measurements:
            return [], []  # If no detections, return empty lists
        
        measurements = np.array(list(measurements))
        centroids = self.quantization_centers.flatten()

        # Compute the nearest centroid for each measurement
        nearest_indices = np.argmin(np.abs(measurements[:, None] - centroids), axis=1)
        quantized_values = centroids[nearest_indices]

        return nearest_indices, quantized_values
    
    def get_positions_from_indices(self, indices):
        """ Returns the quantized positions corresponding to the given quantized indices. """
        indices = np.asarray(indices)  # Ensure indices is a NumPy array
        return self.quantization_centers.flatten()[indices]

    def visualize_quantization(self):
        """ Visualizes the quantization grid and active sensor measurements. """
        plt.figure(figsize=(6,6))

        # Plot quantization grid
        plt.scatter(self.quantization_centers.real, self.quantization_centers.imag, 
                    c='gray', marker='s', s=20, label="Quantization Centers")

        # Plot active sensor detections
        for sensor in self.active_sensors:
            original_positions = np.array(list(sensor.detected_targets.values()))
            quantized_positions = np.array(sensor.quantized_data)

            plt.scatter(original_positions.real, original_positions.imag, c='blue', s=10, label="Original Detections" if sensor.id == self.active_sensors[0].id else "")
            plt.scatter(quantized_positions.real, quantized_positions.imag, c='red', marker='x', s=30, label="Quantized Positions" if sensor.id == self.active_sensors[0].id else "")

        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        plt.title("Quantization Grid and Mapped Positions")
        plt.ylim([-self.area_side/2 - self.area_side/20, self.area_side/2 + self.area_side/20])
        plt.xlim([-self.area_side/2 - self.area_side/20, self.area_side/2 + self.area_side/20])

        plt.hlines([-self.area_side/2, self.area_side/2], xmin=-self.area_side/2, xmax=self.area_side/2, colors="black", linewidth=2)
        plt.vlines([-self.area_side/2, self.area_side/2], ymin=-self.area_side/2, ymax=self.area_side/2, colors="black", linewidth=2)
        plt.ylim([-self.area_side*0.65, self.area_side*0.65])
        plt.xlim([-self.area_side*0.65, self.area_side*0.65])

        plt.grid(True)
        plt.show()
