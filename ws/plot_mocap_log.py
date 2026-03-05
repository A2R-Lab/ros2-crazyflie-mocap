#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def find_latest_mocap_log(directory: Path):
    logs = sorted(
        directory.glob("mocap_log_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return logs[0] if logs else None


def load_mocap_csv(csv_path: Path):
    data = {
        "time_s": [],
        "x": [],
        "y": [],
        "z": [],
        "sp_x": [],
        "sp_y": [],
        "sp_z": [],
    }
    has_setpoint = False

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"time_ms", "x", "y", "z"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(
                f"CSV must contain columns {sorted(required)}, got {reader.fieldnames}"
            )
        has_setpoint = {"sp_x", "sp_y", "sp_z"}.issubset(reader.fieldnames or set())

        for row in reader:
            data["time_s"].append(float(row["time_ms"]) / 1000.0)
            data["x"].append(float(row["x"]))
            data["y"].append(float(row["y"]))
            data["z"].append(float(row["z"]))
            if has_setpoint:
                data["sp_x"].append(float(row["sp_x"]))
                data["sp_y"].append(float(row["sp_y"]))
                data["sp_z"].append(float(row["sp_z"]))

    if not data["time_s"]:
        raise ValueError(f"No data rows found in {csv_path}")

    return data, has_setpoint


def main():
    script_dir = Path(__file__).resolve().parent
    latest_log = find_latest_mocap_log(script_dir)
    parser = argparse.ArgumentParser(description="Plot Crazyflie mocap log CSV data.")
    parser.add_argument(
        "--input",
        type=Path,
        default=latest_log,
        help="Path to mocap CSV log file (default: latest mocap_log_*.csv in script directory)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / (str(latest_log) + "_plot.png"),
        help="Path to output PNG file",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show interactive plot window",
    )
    args = parser.parse_args()
    if args.input is None:
        raise FileNotFoundError(
            f"No mocap_log_*.csv found in {script_dir}. Provide --input explicitly."
        )

    data, has_setpoint = load_mocap_csv(args.input)
    time_s = data["time_s"]
    x = data["x"]
    y = data["y"]
    z = data["z"]

    fig = plt.figure(figsize=(12, 9))
    ax_x = fig.add_subplot(2, 2, 1)
    ax_y = fig.add_subplot(2, 2, 2)
    ax_z = fig.add_subplot(2, 2, 3)
    ax_3d = fig.add_subplot(2, 2, 4, projection="3d")

    all_xyz = x + y + z
    if has_setpoint:
        all_xyz += data["sp_x"] + data["sp_y"] + data["sp_z"]
    xyz_min = min(all_xyz)
    xyz_max = max(all_xyz)
    xyz_pad = max((xyz_max - xyz_min) * 0.05, 1e-3)
    y_min = xyz_min - xyz_pad
    y_max = xyz_max + xyz_pad

    ax_x.plot(time_s, x, color="tab:blue", label="mocap")
    if has_setpoint:
        ax_x.plot(time_s, data["sp_x"], color="tab:orange", linestyle="--", label="setpoint")
    ax_x.set_title("X vs Time")
    ax_x.set_xlabel("Time [s]")
    ax_x.set_ylabel("X [m]")
    ax_x.set_ylim(y_min, y_max)
    ax_x.grid(True)
    if has_setpoint:
        ax_x.legend()

    ax_y.plot(time_s, y, color="tab:green", label="mocap")
    if has_setpoint:
        ax_y.plot(time_s, data["sp_y"], color="tab:orange", linestyle="--", label="setpoint")
    ax_y.set_title("Y vs Time")
    ax_y.set_xlabel("Time [s]")
    ax_y.set_ylabel("Y [m]")
    ax_y.set_ylim(y_min, y_max)
    ax_y.grid(True)
    if has_setpoint:
        ax_y.legend()

    ax_z.plot(time_s, z, color="tab:red", label="mocap")
    if has_setpoint:
        ax_z.plot(time_s, data["sp_z"], color="tab:orange", linestyle="--", label="setpoint")
    ax_z.set_title("Z vs Time")
    ax_z.set_xlabel("Time [s]")
    ax_z.set_ylabel("Z [m]")
    ax_z.set_ylim(y_min, y_max)
    ax_z.grid(True)
    if has_setpoint:
        ax_z.legend()

    ax_3d.plot(x, y, z, color="tab:blue", linewidth=1.5, label="mocap")
    if has_setpoint:
        ax_3d.plot(
            data["sp_x"],
            data["sp_y"],
            data["sp_z"],
            color="tab:orange",
            linewidth=1.2,
            linestyle="--",
            label="setpoint",
        )
    ax_3d.set_title("3D Trajectory")
    ax_3d.set_xlabel("X [m]")
    ax_3d.set_ylabel("Y [m]")
    ax_3d.set_zlabel("Z [m]")
    x_vals = x + (data["sp_x"] if has_setpoint else [])
    y_vals = y + (data["sp_y"] if has_setpoint else [])
    z_vals = z + (data["sp_z"] if has_setpoint else [])
    center_x = (max(x_vals) + min(x_vals)) / 2.0
    center_y = (max(y_vals) + min(y_vals)) / 2.0
    center_z = (max(z_vals) + min(z_vals)) / 2.0
    half_range = max(
        (max(x_vals) - min(x_vals)) / 2.0,
        (max(y_vals) - min(y_vals)) / 2.0,
        (max(z_vals) - min(z_vals)) / 2.0,
    )
    if half_range < 1e-3:
        half_range = 1e-3
    ax_3d.set_xlim(center_x - half_range, center_x + half_range)
    ax_3d.set_ylim(center_y - half_range, center_y + half_range)
    ax_3d.set_zlim(center_z - half_range, center_z + half_range)
    ax_3d.set_box_aspect((1, 1, 1))
    if has_setpoint:
        ax_3d.legend()

    fig.suptitle(f"Mocap Log: {args.input.name}", fontsize=12)
    plt.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Plot saved to {args.output}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
