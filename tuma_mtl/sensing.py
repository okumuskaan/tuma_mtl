"""
sensing.py

Author: Kaan Okumus
Date: March 2025

This module defines the **Sensing Environment** for multi-target localization (MTL). It models how sensors detect targets 
based on probabilistic detection functions and allows visualization of the sensing process.

### Key Components:
- **Marcum Q-Function (`marcumq`)**: Computes the probability of detection given signal parameters.
- **SensingEnvironment**: Manages sensors, targets, and detection interactions.
  - Generates and initializes sensors and targets in a defined area.
  - Computes detection probabilities based on distance.
  - Runs the sensing process for all sensors.
  - Provides statistics and visualization tools for detections.

### Main Functionalities:
- **`create_P_d_function()`**: Defines the probability of detection (`P_d`) based on sensor parameters.
- **`compute_sensing_radius()`**: Computes the effective sensing range.
- **`generate_sensors()` and `generate_targets()`**: Places sensors and targets in the environment.
- **`run_sensing()`**: Simulates the detection process.
- **`get_detection_statistics()`**: Provides insights into how many sensors and targets were active.
- **Visualization Functions**:
  - `visualize_initial_setup()`: Shows the initial setup of sensors and targets.
  - `visualize()` and `visualize_active()`: Displays detected sensors and targets.
  - `visualize_detection_probability()`: Plots the detection probability curve.
  - `visualize_approx_sensing_radius_vs_Ns()`: Shows how the sensing range varies with sensing blocklength.

### Usage:
This module is primarily used in **MTL applications**, where the detection probability of targets is an important factor.
"""

import scipy.stats as stats # type: ignore
from scipy.optimize import bisect # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore

from tuma_mtl.devices import *

def marcumq(a, b, m=1):
    """Computes the generalized Marcum Q-function."""
    return stats.ncx2.sf(b**2, 2*m, a**2)

class SensingEnvironment:
    """ Manages sensors, targets, and detection interactions. """
    def __init__(self, num_sensors, num_targets, area_side, sigma_noise, perfect_measurement=False, max_detections_per_sensor=None,
                 Ns=2000, Ps=0.0009765625, sigma_n=10**(-4), S=100, fc=2.8e9, gamma=13.815510557964274):
        """
        Initializes the sensing environment.
        
        Parameters:
        - num_sensors: Number of sensors
        - num_targets: Number of targets
        - area_side: Side length of the area (square region)
        - sigma_noise: Standard deviation of measurement noise
        - perfect_measurement: If True, sensors measure without noise
        - max_detections_per_sensor: Maximum detections per sensor
        - Ns: Number of sensing blocklengths
        - Ps: Sensing power
        - sigma_n: Noise power
        - S: Target cross-section area
        - fc: Carrier frequency
        - gamma: Detection threshold parameter
        """
        self.area_side = area_side
        self.Ns = Ns
        self.Ps = Ps
        self.sigma_n = sigma_n
        self.S = S
        self.fc = fc
        self.gamma = gamma

        # Define detection probability function for this environment
        self.P_d_func = self.create_P_d_function()

        # Compute the sensing radius for visualization
        sensing_radius = self.compute_sensing_radius(0.01)

        # Generate sensors and targets
        self.sensors = self.generate_sensors(num_sensors, sigma_noise, sensing_radius, perfect_measurement, max_detections_per_sensor)
        self.targets = self.generate_targets(num_targets)

    def create_P_d_function(self):
        """ Returns a function that computes the detection probability based on distance. """
        c = 3 * 10**8  # Speed of light in m/s
        lambda_ = c / self.fc  # Wavelength

        def P_d(dist):
            return marcumq(np.sqrt((2 * self.Ns * self.Ps * self.S * (lambda_**2)) /
                                   (((4 * np.pi)**3) * (dist**4) * (self.sigma_n**2))),
                           np.sqrt(self.gamma))

        return P_d
    
    def compute_sensing_radius(self, cutoff_prob=0.2):
        """ Computes the least distance where P_d < cutoff_prob for visualization. """
        def detection_threshold(d):
            return self.P_d_func(d) - cutoff_prob
        try:
            radius = bisect(detection_threshold, 1, 5000)  # Search for a reasonable range (1m to 5000m)
        except ValueError:
            radius = 1000  # Default fallback if no solution found
        return radius

    def generate_positions(self, N):
        """ Generate N uniformly distributed positions in the area. """
        x_coords = np.random.uniform(-self.area_side/2, self.area_side/2, N)
        y_coords = np.random.uniform(-self.area_side/2, self.area_side/2, N)
        return x_coords + 1j * y_coords  # Return complex coordinates

    def generate_sensors(self, num_sensors, sigma_noise, sensing_radius, perfect_measurement=False, max_detections_per_sensor=None):
        """ Generate sensor objects with random positions. """
        positions = self.generate_positions(num_sensors)
        return [Sensor(sensor_id=i, position=pos, sigma_noise=sigma_noise, P_d_func=self.P_d_func, sensing_radius=sensing_radius,
                    perfect_measurement=perfect_measurement, max_detections_per_sensor=max_detections_per_sensor) 
                for i, pos in enumerate(positions)]
    
    def generate_targets(self, num_targets):
        """ Generate target objects with random positions. """
        positions = self.generate_positions(num_targets)
        return [Target(target_id=i, position=pos) for i, pos in enumerate(positions)]

    def run_sensing(self):
        """ Execute sensing process for all sensors. """
        for sensor in self.sensors:
            sensor.detect_targets(self.targets)

    def get_targets_and_active_sensors(self):
        """ Returns lists of active sensor and target objects. """
        active_sensors = [sensor for sensor in self.sensors if sensor.is_active]        
        return self.targets, active_sensors

    def get_detection_statistics(self):
        """ Returns statistics on sensor detections, including unique active targets. """
        detected_counts = [sensor.num_detected for sensor in self.sensors]
        active_sensors = sum(sensor.is_active for sensor in self.sensors)
        total_detections = sum(detected_counts)  # Sum of all detections (can be > unique targets)
        avg_detections = np.mean(detected_counts)
        max_detections = np.max(detected_counts)

        # Compute number of active (detected at least once) targets
        active_targets = sum(target.is_active for target in self.targets)

        return {
            "Total Detections": total_detections,  # Includes duplicate detections
            "Total Active Targets": active_targets,  # Unique targets detected at least once
            "Percentage of Active Targets": (active_targets / len(self.targets)) * 100,
            "Average Detections per Sensor": avg_detections,
            "Max Detections by a Sensor": max_detections,
            "Number of Active Sensors": active_sensors,
            "Percentage of Active Sensors": (active_sensors / len(self.sensors)) * 100
        }

    def visualize_initial_setup(self):
        """ Visualize the initial sensors and targets. """
        plt.figure(figsize=(6,6))
        
        # Plot targets
        target_positions = np.array([target.position for target in self.targets])
        plt.scatter(target_positions.real, target_positions.imag, c='red', marker='x', label="Targets")

        # Plot sensors
        sensor_positions = np.array([target.position for target in self.sensors])
        plt.scatter(sensor_positions.real, sensor_positions.imag, c='blue', s=10, label="Sensors")

        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        plt.title(f"Sensing Environment\nwith K = {len(self.sensors)} sensors and T = {len(self.targets)} targets")
        
        plt.hlines([-self.area_side/2, self.area_side/2], xmin=-self.area_side/2, xmax=self.area_side/2, colors="black", linewidth=2)
        plt.vlines([-self.area_side/2, self.area_side/2], ymin=-self.area_side/2, ymax=self.area_side/2, colors="black", linewidth=2)
        plt.ylim([-self.area_side*0.65, self.area_side*0.65])
        plt.xlim([-self.area_side*0.65, self.area_side*0.65])

        plt.grid(True)
        plt.show()

    def visualize(self):
        """ Visualize the sensors, targets, and sensing regions. """
        plt.figure(figsize=(6,6))
        
        # Plot targets
        target_positions = np.array([target.position for target in self.targets])
        plt.scatter(target_positions.real, target_positions.imag, c='red', marker='x', label="Targets")

        # Plot sensors
        active_sensors = [sensor.position for sensor in self.sensors if sensor.is_active]
        inactive_sensors = [sensor.position for sensor in self.sensors if not sensor.is_active]

        if active_sensors:
            plt.scatter([s.real for s in active_sensors], [s.imag for s in active_sensors], c='blue', s=10, label="Active Sensors")
        if inactive_sensors:
            plt.scatter([s.real for s in inactive_sensors], [s.imag for s in inactive_sensors], c='gray', s=10, label="Inactive Sensors", alpha=0.5)

        # Draw sensing circles for active sensors
        for sensor in self.sensors:
            if sensor.is_active:
                circle = plt.Circle((sensor.position.real, sensor.position.imag), sensor.sensing_radius, color='gray', alpha=0.2)
                plt.gca().add_patch(circle)

        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        plt.title(f"Sensing Environment\nwith K = {len(self.sensors)} sensors and T = {len(self.targets)} targets")
        
        plt.hlines([-self.area_side/2, self.area_side/2], xmin=-self.area_side/2, xmax=self.area_side/2, colors="black", linewidth=2)
        plt.vlines([-self.area_side/2, self.area_side/2], ymin=-self.area_side/2, ymax=self.area_side/2, colors="black", linewidth=2)
        plt.ylim([-self.area_side*0.65, self.area_side*0.65])
        plt.xlim([-self.area_side*0.65, self.area_side*0.65])

        plt.grid(True)
        plt.show()

    def visualize_active(self):
        """ Visualize only active sensors and active targets with sensing circles. """
        plt.figure(figsize=(6,6))

        # Get positions of active targets
        active_targets = [target.position for target in self.targets if target.is_active]

        # Get positions of active sensors
        active_sensors = [sensor.position for sensor in self.sensors if sensor.is_active]

        # Plot active targets
        if active_targets:
            plt.scatter([t.real for t in active_targets], [t.imag for t in active_targets], 
                        c='red', marker='x', label="Active Targets")

        # Plot active sensors and their sensing areas
        if active_sensors:
            plt.scatter([s.real for s in active_sensors], [s.imag for s in active_sensors], 
                        c='blue', s=10, label="Active Sensors")

            # Draw sensing circles
            for sensor in self.sensors:
                if sensor.is_active:
                    circle = plt.Circle((sensor.position.real, sensor.position.imag), 
                                        sensor.sensing_radius, color='gray', alpha=0.2)
                    plt.gca().add_patch(circle)

        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        plt.title(f"Active Sensors and Targets with Sensing Circles\nwith Ka = {len(active_sensors)} active sensors and Ta = {len(active_targets)} active targets")
        plt.ylim([-self.area_side/2 - self.area_side/20, self.area_side/2 + self.area_side/20])
        plt.xlim([-self.area_side/2 - self.area_side/20, self.area_side/2 + self.area_side/20])

        plt.hlines([-self.area_side/2, self.area_side/2], xmin=-self.area_side/2, xmax=self.area_side/2, colors="black", linewidth=2)
        plt.vlines([-self.area_side/2, self.area_side/2], ymin=-self.area_side/2, ymax=self.area_side/2, colors="black", linewidth=2)
        plt.ylim([-self.area_side*0.65, self.area_side*0.65])
        plt.xlim([-self.area_side*0.65, self.area_side*0.65])

        plt.grid(True)
        plt.show()

    def visualize_detection_probability(self, figsize=(10,5), fig=None):
        """ Visualizes the detection probability in sensing versus distances. """
        dists = np.linspace(0.01,100,10000)
        Pds = self.P_d_func(dists)
        sensing_radius = self.compute_sensing_radius(1e-2)

        if fig is None:
            plt.figure(figsize=figsize)
            plt.semilogy(dists, Pds, linewidth=2.0)
        else:
            plt.semilogy(dists, Pds, label=f"N_s = {self.Ns}", linewidth=2.0)
        #plt.vlines([sensing_radius], 0, 1, "red", "dashed")
        #plt.text(sensing_radius+2, 0.3, f"Sensing radius: {sensing_radius:.2f} m", color="red", rotation=90)
        plt.title("Detection Probability")
        plt.xlabel("Distance (m)")
        plt.ylabel("P_d")
        plt.grid(True)
        if fig is None:
            plt.show()

    def visualize_approx_sensing_radius_vs_Ns(self, figsize=(10,5)):
        c = 3 * 10**8  # Speed of light in m/s
        lambda_ = c / self.fc  # Wavelength

        cutoff_prob = 0.01

        N_ss = np.linspace(1,3000,100).astype(int)
        sensing_radius_s = np.zeros(len(N_ss))

        for idx_Ns, N_s in enumerate(N_ss):
            def P_d(dist):
                return marcumq(np.sqrt((2 * N_s * self.Ps * self.S * (lambda_**2)) /
                                    (((4 * np.pi)**3) * (dist**4) * (self.sigma_n**2))),
                            np.sqrt(self.gamma))
            def detection_threshold(d):
                return P_d(d) - cutoff_prob
            
            try:
                sensing_radius = bisect(detection_threshold, 1, 5000)
            except ValueError:
                sensing_radius = 10000
            sensing_radius_s[idx_Ns] = sensing_radius            

        plt.figure(figsize=figsize)
        plt.semilogx(N_ss, sensing_radius_s)
        plt.title("Approximated Sensing Radius vs Sensing Blocklength")
        plt.ylabel("Approximated sensing radius (m)")
        plt.xlabel("Sensing Blocklength, N_s")
        plt.grid(True)
        plt.show()

    def visualize_alternative(self):
        """ Visualizes the sensors with sensing regions based on detection probabilities. """
        plt.subplots(figsize=(6, 6))

        # Create a grid for visualization
        grid_size = 200
        x = np.linspace(-self.area_side / 2, self.area_side / 2, grid_size)
        y = np.linspace(-self.area_side / 2, self.area_side / 2, grid_size)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for sensor in self.sensors:
            distances = np.sqrt((X - sensor.position.real)**2 + (Y - sensor.position.imag)**2)
            Z = np.maximum(Z, self.P_d_func(distances*1000)) # Take max probability at each point

        # Normalize to keep the contrast clear
        Z = 1 - Z

        # Plot the sensing probability heatmap
        plt.imshow(Z, extent=[-self.area_side/2, self.area_side/2, -self.area_side/2, self.area_side/2], 
                origin='lower', cmap="gray", alpha=0.5)
        
        # Plot targets
        active_targets = [target.position for target in self.targets if target.is_active]
        inactive_targets = [target.position for target in self.targets if not target.is_active]
        if active_targets:
            plt.scatter([t.real for t in active_targets], [t.imag for t in active_targets], c='red', marker="x", label="Active Targets")
        if inactive_targets:
            plt.scatter([t.real for t in inactive_targets], [t.imag for t in inactive_targets], c='darkgray', marker="x", alpha=0.8, label="Inactive Targets")

        # Plot sensors
        active_sensors = [sensor.position for sensor in self.sensors if sensor.is_active]
        inactive_sensors = [sensor.position for sensor in self.sensors if not sensor.is_active]
        if active_sensors:
            plt.scatter([s.real for s in active_sensors], [s.imag for s in active_sensors], c='blue', s=10, label="Active Sensors")
        if inactive_sensors:
            plt.scatter([s.real for s in inactive_sensors], [s.imag for s in inactive_sensors], c='black', s=10, label="Inactive Sensors", alpha=1)

        # Formatting
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        plt.title(f"Sensing Environment\nwith K = {len(self.sensors)} sensors and T = {len(self.targets)} targets")    
        plt.hlines([-self.area_side/2, self.area_side/2], xmin=-self.area_side/2, xmax=self.area_side/2, colors="black", linewidth=2)
        plt.vlines([-self.area_side/2, self.area_side/2], ymin=-self.area_side/2, ymax=self.area_side/2, colors="black", linewidth=2)
        plt.ylim([-self.area_side*0.65, self.area_side*0.65])
        plt.xlim([-self.area_side*0.65, self.area_side*0.65])
        plt.grid(True)
        plt.show()

    def visualize_active_alternative(self):
        """ Visualize only active sensors and active targets with sensing circles. """
        plt.figure(figsize=(6,6))

        # Get positions of active targets
        active_targets = [target.position for target in self.targets if target.is_active]

        # Get positions of active sensors
        active_sensors = [sensor.position for sensor in self.sensors if sensor.is_active]

        # Plot active targets
        if active_targets:
            plt.scatter([t.real for t in active_targets], [t.imag for t in active_targets], 
                        c='red', marker='x', label="Active Targets")
            
        # Create a grid for visualization
        grid_size = 200
        x = np.linspace(-self.area_side / 2, self.area_side / 2, grid_size)
        y = np.linspace(-self.area_side / 2, self.area_side / 2, grid_size)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        # Plot active sensors and their sensing areas
        if active_sensors:
            plt.scatter([s.real for s in active_sensors], [s.imag for s in active_sensors], 
                        c='blue', s=10, label="Active Sensors")

            # Draw sensing circles
            for sensor in self.sensors:
                if sensor.is_active:
                    distances = np.sqrt((X - sensor.position.real)**2 + (Y - sensor.position.imag)**2)
                    Z = np.maximum(Z, self.P_d_func(distances*1000)) # Take max probability at each point

        # Normalize to keep the contrast clear
        Z = 1 - Z

        # Plot the sensing probability heatmap
        plt.imshow(Z, extent=[-self.area_side/2, self.area_side/2, -self.area_side/2, self.area_side/2], 
                origin='lower', cmap="gray", alpha=0.5)

        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        plt.title(f"Active Sensors and Targets with Sensing Circles\nwith Ka = {len(active_sensors)} active sensors and Ta = {len(active_targets)} active targets")
        plt.ylim([-self.area_side/2 - self.area_side/20, self.area_side/2 + self.area_side/20])
        plt.xlim([-self.area_side/2 - self.area_side/20, self.area_side/2 + self.area_side/20])

        plt.hlines([-self.area_side/2, self.area_side/2], xmin=-self.area_side/2, xmax=self.area_side/2, colors="black", linewidth=2)
        plt.vlines([-self.area_side/2, self.area_side/2], ymin=-self.area_side/2, ymax=self.area_side/2, colors="black", linewidth=2)
        plt.ylim([-self.area_side*0.65, self.area_side*0.65])
        plt.xlim([-self.area_side*0.65, self.area_side*0.65])

        plt.grid(True)
        plt.show()
