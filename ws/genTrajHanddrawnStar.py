#!/usr/bin/env python3

import csv
import numpy as np
import matplotlib.pyplot as plt


csv_filename = "trajectory.csv"
plot_filename = "trajectory.png"

# Parameters
outer_radius = 0.3
z_height = 0.0
duration = 10.0  # seconds
yaw = 0.0
# Regular pentagon outer vertices (clockwise), with top at angle 90 deg
angles = np.deg2rad([90, 18, -54, -126, 162])
outer = [
    [outer_radius * np.cos(a), outer_radius * np.sin(a), z_height]
    for a in angles
]

# Label vertices:
# A=top, B=right-upper, C=right-lower, D=left-lower, E=left-upper
A, B, C, D, E = outer

# Hand-drawn order:
# 1) upside-down V: D -> A -> C
# 2) horizontal cross: C -> E -> B (segment E->B is the horizontal line)
# 3) connect ends: B -> D
stroke_points = [D, A, C, E, B, D]
trajectory_xyz = stroke_points

# Build CSV rows: [timestamp_ms, x, y, z, yaw]
timestamps = np.linspace(0.0, duration * 1000.0, len(trajectory_xyz))
trajectory_rows = []
for t_ms, p in zip(timestamps, trajectory_xyz):
    trajectory_rows.append([t_ms, p[0], p[1], p[2], yaw])

# Save CSV
with open(csv_filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "x", "y", "z", "yaw"])
    writer.writerows(trajectory_rows)

# Plot
xs = [p[0] for p in trajectory_xyz]
ys = [p[1] for p in trajectory_xyz]
zs = [p[2] for p in trajectory_xyz]

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot(xs, ys, zs, "-", color="orange")
ax.scatter(
    [p[0] for p in stroke_points],
    [p[1] for p in stroke_points],
    [p[2] for p in stroke_points],
    color="black",
    s=20,
)
ax.set_title("Hand-Drawn Star Trajectory (3D)")
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")

# Equal scale on x/y/z
xs_arr = np.array(xs)
ys_arr = np.array(ys)
zs_arr = np.array(zs)
center_x = (xs_arr.max() + xs_arr.min()) / 2
center_y = (ys_arr.max() + ys_arr.min()) / 2
center_z = (zs_arr.max() + zs_arr.min()) / 2
half_range = max(
    (xs_arr.max() - xs_arr.min()) / 2,
    (ys_arr.max() - ys_arr.min()) / 2,
    (zs_arr.max() - zs_arr.min()) / 2,
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
