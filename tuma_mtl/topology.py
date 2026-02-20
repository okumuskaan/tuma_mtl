"""
topology.py

Author: Kaan Okumus  
Date: March 2025  

This module defines the **Network Topology** for the TUMA system, organizing zones and access points (APs).  
It provides methods to generate a structured network, assign zones, and position APs,  
as well as visualization tools to display the network layout.

### Key Components:
- **Zone**: Represents a communication zone in the network.
  - Each zone has a unique ID, a center position, and a defined side length.
  - Provides encoding (`Cx`) and decoding (`Cy`) operations.
  - Stores blocklength, codebook size, and codebook matrix (`C`) for encoding.

- **NetworkTopology**: Manages the overall network structure.
  - Defines the system layout with **U zones** in a grid.
  - Generates **Access Points (APs)** positioned along zone boundaries.
  - Assigns each sensor to its corresponding zone based on location.
  - Supports visualization of zones and AP positions.

### Main Functionalities:
- **`generate_zones()`**: Creates a structured grid of zone centers.
- **`generate_aps()`**: Generates and places APs on the boundaries between zones.
- **`get_zone_by_position(position)`**: Determines the zone ID for a given position.
- **`visualize()`**: Displays the network topology, including zones and APs.

### Usage:
This module is used for defining and managing the topology of the **TUMA communication system**,  
ensuring structured communication zones and well-placed APs for decoding purposes.
"""

import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore

from tuma_mtl.devices import AccessPoint

class Zone:
    """ Represents a square-shaped zone in the system. """
    def __init__(self, zone_id, center_position, side_length):
        """
        Parameters:
        - zone_id: Unique ID of the zone.
        - center_position: Complex coordinate of the zone center.
        - side_length: Side length of the zone.
        """
        self.id = zone_id
        self.center = center_position
        self.side_length = side_length
        
        # Will be set later inside TUMAEnvironment
        self.blocklength = None
        self.codebook_size = None
        self.C = None  

    def Cx(self, x):
        """ Applies encoding (Cx) operation. """
        if self.C is None:
            raise ValueError(f"Zone {self.id}: Codebook C is not initialized.")
        return self.C @ x  # Matrix-vector multiplication

    def Cy(self, y):
        """ Applies decoding (Cy) operation. """
        if self.C is None:
            raise ValueError(f"Zone {self.id}: Codebook C is not initialized.")
        return self.C.conj().T @ y  # Hermitian transpose

class NetworkTopology:
    """ Manages the system topology, including zones and APs. """
    def __init__(self, side_length, rows=3, cols=3, jitter=0):
        self.side_length = side_length
        self.area_side = side_length*3
        self.rows = rows
        self.cols = cols
        self.U = rows * cols  # Number of zones
        self.B = None  # Number of APs (set after generating APs)
        self.jitter = jitter  # Random perturbation for AP positions

        self.zones = self.generate_zones()
        self.aps = self.generate_aps()

    def generate_zones(self):
        """ Generate zone centroids for a 3x3 grid. """
        zone_centers_x = np.arange(self.cols) - (self.cols - 1) / 2
        zone_centers_y = np.arange(self.rows) - (self.rows - 1) / 2
        zone_centers = np.sum([ar.astype(float) * (1 - i + 1j * i) for i, ar in enumerate(np.meshgrid(zone_centers_x, zone_centers_y))], axis=0) * self.side_length
        
        # Create Zone objects
        zones = [Zone(zone_id=i, center_position=center, side_length=self.side_length) 
                 for i, center in enumerate(zone_centers.flatten())]
        return zones

    def generate_aps(self):
        """ Generate APs positioned on zone boundaries, with extra APs between each adjacent AP. """
        aps_set = set()  # Use a set to ensure uniqueness

        # Horizontal AP positions (between rows)
        for row in range(self.rows + 1):
            y_pos = (row - self.rows / 2) * self.side_length
            x_positions = np.linspace(-self.cols/2 * self.side_length, self.cols/2 * self.side_length, 2*self.cols + 1)
            for x in x_positions:
                aps_set.add(complex(round(x, 6), round(y_pos, 6)))  # Round to avoid near-duplicates

        # Vertical AP positions (between columns)
        for col in range(self.cols + 1):
            x_pos = (col - self.cols / 2) * self.side_length
            y_positions = np.linspace(-self.rows/2 * self.side_length, self.rows/2 * self.side_length, 2*self.rows + 1)
            for y in y_positions:
                aps_set.add(complex(round(x_pos, 6), round(y, 6)))  # Round to avoid near-duplicates

        # Convert set to list
        aps_positions = sorted(aps_set, key=lambda p: (p.real, p.imag))

        # Create AP objects
        aps = [AccessPoint(ap_id=i, position=pos) for i, pos in enumerate(aps_positions)]
        self.B = len(aps)  # Store number of APs
        return aps

    def get_zone_by_position(self, position):
        """ Determines which zone a given position belongs to. """
        for zone in self.zones:
            x_min = zone.center.real - zone.side_length / 2
            x_max = zone.center.real + zone.side_length / 2
            y_min = zone.center.imag - zone.side_length / 2
            y_max = zone.center.imag + zone.side_length / 2

            if x_min <= position.real < x_max and y_min <= position.imag < y_max:
                return zone.id
        return None  # Out of bounds

    def visualize(self):
        """ Visualizes zones and APs. """
        plt.figure(figsize=(6, 6))

        # Plot Zones
        zone_positions = np.array([zone.center for zone in self.zones])
        plt.scatter(zone_positions.real, zone_positions.imag, c='blue', marker='s', s=100, label="Zones (Centers)")

        # Plot APs
        ap_positions = np.array([ap.position for ap in self.aps])
        plt.scatter(ap_positions.real, ap_positions.imag, c='orange', marker='o', label="Access Points (APs)", s=100)

        # Draw zone boundaries
        for zone in self.zones:
            rect = plt.Rectangle((zone.center.real - zone.side_length / 2, zone.center.imag - zone.side_length / 2),
                                 zone.side_length, zone.side_length, linewidth=1, edgecolor='gray', facecolor='none')
            plt.gca().add_patch(rect)

        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        plt.title("Network Topology (Zones and APs)")
        plt.ylim([-self.rows*self.side_length/2 - self.rows*self.side_length/20, self.rows*self.side_length/2 + self.rows*self.side_length/20])
        plt.xlim([-self.cols*self.side_length/2 - self.cols*self.side_length/20, self.cols*self.side_length/2 + self.cols*self.side_length/20])
        
        plt.hlines([-self.area_side/2, self.area_side/2], xmin=-self.area_side/2, xmax=self.area_side/2, colors="black", linewidth=2)
        plt.vlines([-self.area_side/2, self.area_side/2], ymin=-self.area_side/2, ymax=self.area_side/2, colors="black", linewidth=2)
        plt.ylim([-self.area_side*0.65, self.area_side*0.65])
        plt.xlim([-self.area_side*0.65, self.area_side*0.65])
        
        plt.grid(True)
        plt.show()
