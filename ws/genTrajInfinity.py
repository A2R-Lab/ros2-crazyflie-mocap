#!/usr/bin/env python3

import csv
import numpy as np
import matplotlib.pyplot as plt


csv_filename = "trajectory.csv"
plot_filename = "trajectory.png"

# Parameters
radius = 0.6         # horizontal size of the infinity loop [m]
z_height = 0.0       # constant altitude [m]
duration =  8.0      # total duration [s]
num_points = 200     # number of setpoints
yaw = 0.0            # constant yaw [deg]

# Figure-8 (Gerono lemniscate): x = a*sin(t), y = a*sin(t)*cos(t)
t = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
y = radius * np.sin(t)
x = radius * np.sin(t) * np.cos(t)
z = np.full_like(x, z_height)

# CSV rows: [timestamp_ms, x, y, z, yaw]
timestamps = np.linspace(0.0, duration * 1000.0, num_points)
trajectory_rows = [[ts, px, py, pz, yaw] for ts, px, py, pz in zip(timestamps, x, y, z)]

with open(csv_filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "x", "y", "z", "yaw"])
    writer.writerows(trajectory_rows)

# Plot (closed loop for display)
x_plot = np.append(x, x[0])
y_plot = np.append(y, y[0])
z_plot = np.append(z, z[0])

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot(x_plot, y_plot, z_plot, "o-", color="orange", markersize=2)
ax.set_title("Infinity Trajectory (3D)")
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")

# Equal scale on x/y/z
center_x = (x_plot.max() + x_plot.min()) / 2.0
center_y = (y_plot.max() + y_plot.min()) / 2.0
center_z = (z_plot.max() + z_plot.min()) / 2.0
half_range = max(
    (x_plot.max() - x_plot.min()) / 2.0,
    (y_plot.max() - y_plot.min()) / 2.0,
    (z_plot.max() - z_plot.min()) / 2.0,
)
if half_range < 1e-3:
    half_range = 1e-3
ax.set_xlim(center_x - half_range, center_x + half_range)
ax.set_ylim(center_y - half_range, center_y + half_range)
ax.set_zlim(center_z - half_range, center_z + half_range)
ax.set_box_aspect((1, 1, 1))
ax.grid(True)

plt.tight_layout()
plt.savefig(plot_filename, dpi=150)
plt.close(fig)

print("CSV saved as", csv_filename)
print("Plot saved as", plot_filename)
