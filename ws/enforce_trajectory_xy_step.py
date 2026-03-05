#!/usr/bin/env python3

import argparse
import csv
import math
from pathlib import Path


def wrap_deg(angle):
    """Wrap angle to [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


def interp_yaw_shortest(y0, y1, alpha):
    """Interpolate yaw in degrees using shortest angular distance."""
    delta = wrap_deg(y1 - y0)
    return wrap_deg(y0 + alpha * delta)


def load_rows(path: Path):
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append(
                {
                    "timestamp": float(row["timestamp"]),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "z": float(row["z"]),
                    "yaw": float(row["yaw"]),
                }
            )
    if len(rows) < 2:
        raise ValueError(f"Need at least 2 trajectory points in {path}")
    return rows


def midpoint(a, b):
    return {
        "timestamp": 0.5 * (a["timestamp"] + b["timestamp"]),
        "x": 0.5 * (a["x"] + b["x"]),
        "y": 0.5 * (a["y"] + b["y"]),
        "z": 0.5 * (a["z"] + b["z"]),
        "yaw": interp_yaw_shortest(a["yaw"], b["yaw"], 0.5),
    }


def enforce_max_xy_step(rows, max_xy_step_m):
    new_rows = [rows[0]]
    inserted_points = 0

    def append_with_midpoints(start, end):
        nonlocal inserted_points
        dx = end["x"] - start["x"]
        dy = end["y"] - start["y"]
        dist = math.hypot(dx, dy)

        if dist <= max_xy_step_m:
            new_rows.append(end)
            return

        mid = midpoint(start, end)
        inserted_points += 1
        append_with_midpoints(start, mid)
        append_with_midpoints(mid, end)

    for next_row in rows[1:]:
        append_with_midpoints(new_rows[-1], next_row)

    return new_rows, inserted_points


def max_xy_step(rows):
    max_step = 0.0
    for i in range(1, len(rows)):
        dx = rows[i]["x"] - rows[i - 1]["x"]
        dy = rows[i]["y"] - rows[i - 1]["y"]
        max_step = max(max_step, math.hypot(dx, dy))
    return max_step


def write_rows(path: Path, rows):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "x", "y", "z", "yaw"])
        for row in rows:
            writer.writerow([row["timestamp"], row["x"], row["y"], row["z"], row["yaw"]])


def main():
    parser = argparse.ArgumentParser(
        description="Ensure consecutive XY trajectory steps are below a max distance by interpolation."
    )
    parser.add_argument("--input", default="trajectory.csv", help="Input trajectory CSV path")
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: overwrite --input)",
    )
    parser.add_argument(
        "--max-step",
        type=float,
        default=0.05,
        help="Maximum allowed sqrt((dx)^2 + (dy)^2) in meters (default: 0.20)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path

    rows = load_rows(input_path)
    before = max_xy_step(rows)
    adjusted_rows, inserted = enforce_max_xy_step(rows, args.max_step)
    after = max_xy_step(adjusted_rows)
    write_rows(output_path, adjusted_rows)

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Rows before: {len(rows)}")
    print(f"Rows after:  {len(adjusted_rows)}")
    print(f"Inserted:    {inserted}")
    print(f"Max XY step before: {before:.6f} m")
    print(f"Max XY step after:  {after:.6f} m")
    print(f"Limit:             {args.max_step:.6f} m")


if __name__ == "__main__":
    main()
