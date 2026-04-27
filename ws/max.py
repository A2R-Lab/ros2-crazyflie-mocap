#!/usr/bin/env python3
import argparse
import csv
import glob
import math
import os
from typing import List, Optional, Tuple


def latest_cf_log() -> Optional[str]:
    files = glob.glob("cf_log_*.csv")
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def to_float(value: str) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return v


def max_speed(t_ms: List[float], x: List[float], y: List[float], z: List[float]) -> Tuple[Optional[float], int]:
    best = None
    used = 0
    for i in range(1, len(t_ms)):
        t0, t1 = t_ms[i - 1], t_ms[i]
        x0, x1 = x[i - 1], x[i]
        y0, y1 = y[i - 1], y[i]
        z0, z1 = z[i - 1], z[i]
        if any(math.isnan(v) for v in (t0, t1, x0, x1, y0, y1, z0, z1)):
            continue
        dt = (t1 - t0) / 1000.0
        if dt <= 0.0:
            continue
        ds = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)
        v = ds / dt
        if best is None or v > best:
            best = v
        used += 1
    return best, used


def speed_series(t_ms: List[float], x: List[float], y: List[float], z: List[float]) -> List[float]:
    vals: List[float] = []
    for i in range(1, len(t_ms)):
        t0, t1 = t_ms[i - 1], t_ms[i]
        x0, x1 = x[i - 1], x[i]
        y0, y1 = y[i - 1], y[i]
        z0, z1 = z[i - 1], z[i]
        if any(math.isnan(v) for v in (t0, t1, x0, x1, y0, y1, z0, z1)):
            continue
        dt = (t1 - t0) / 1000.0
        if dt <= 0.0:
            continue
        ds = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)
        vals.append(ds / dt)
    return vals


def median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    data = sorted(values)
    n = len(data)
    mid = n // 2
    if n % 2 == 1:
        return data[mid]
    return 0.5 * (data[mid - 1] + data[mid])


def remove_outliers_mad(values: List[float], z_thresh: float) -> Tuple[List[float], int]:
    if len(values) < 5:
        return values[:], 0
    med = median(values)
    if med is None:
        return values[:], 0
    abs_dev = [abs(v - med) for v in values]
    mad = median(abs_dev)
    if mad is None or mad <= 1e-12:
        return values[:], 0
    # 1.4826 scales MAD to std-equivalent for normal data.
    scale = 1.4826 * mad
    kept: List[float] = []
    removed = 0
    for v in values:
        z = abs(v - med) / scale
        if z <= z_thresh:
            kept.append(v)
        else:
            removed += 1
    return kept, removed


def max_tilt(roll_deg: List[float], pitch_deg: List[float]) -> Tuple[Optional[float], Optional[float], int]:
    best_deg = None
    used = 0
    for r, p in zip(roll_deg, pitch_deg):
        if math.isnan(r) or math.isnan(p):
            continue
        # stabilizer roll/pitch are degrees
        tilt_deg = math.sqrt(r * r + p * p)
        if best_deg is None or tilt_deg > best_deg:
            best_deg = tilt_deg
        used += 1
    if best_deg is None:
        return None, None, 0
    return best_deg, math.radians(best_deg), used


def col_present(fieldnames: List[str], name: str) -> bool:
    return name in fieldnames


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute max speed and tilt from Crazyflie CSV logs.")
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Path to log CSV (default: latest cf_log_*.csv in current directory)",
    )
    parser.add_argument(
        "--outlier-z",
        type=float,
        default=6.0,
        help="MAD z-threshold for rejecting speed outliers (default: 6.0)",
    )
    args = parser.parse_args()

    path = args.input or latest_cf_log()
    if path is None:
        print("No cf_log_*.csv found in current directory.")
        return 1

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            print(f"Empty/invalid CSV: {path}")
            return 1

        required = ["time_ms", "state_x", "state_y", "state_z", "roll", "pitch"]
        missing = [c for c in required if not col_present(reader.fieldnames, c)]
        if missing:
            print(f"Missing required columns in {path}: {', '.join(missing)}")
            return 1

        has_mocap = all(
            col_present(reader.fieldnames, c)
            for c in ("mocap_time_ms", "mocap_x", "mocap_y", "mocap_z")
        )

        t_ms: List[float] = []
        sx: List[float] = []
        sy: List[float] = []
        sz: List[float] = []
        roll: List[float] = []
        pitch: List[float] = []

        tm_ms: List[float] = []
        mx: List[float] = []
        my: List[float] = []
        mz: List[float] = []

        rows = 0
        for row in reader:
            rows += 1
            t_ms.append(to_float(row.get("time_ms", "")))
            sx.append(to_float(row.get("state_x", "")))
            sy.append(to_float(row.get("state_y", "")))
            sz.append(to_float(row.get("state_z", "")))
            roll.append(to_float(row.get("roll", "")))
            pitch.append(to_float(row.get("pitch", "")))
            if has_mocap:
                tm_ms.append(to_float(row.get("mocap_time_ms", "")))
                mx.append(to_float(row.get("mocap_x", "")))
                my.append(to_float(row.get("mocap_y", "")))
                mz.append(to_float(row.get("mocap_z", "")))

    state_max, state_pairs = max_speed(t_ms, sx, sy, sz)
    state_speeds = speed_series(t_ms, sx, sy, sz)
    state_speeds_clean, state_removed = remove_outliers_mad(state_speeds, args.outlier_z)
    state_max_clean = max(state_speeds_clean) if state_speeds_clean else None
    tilt_deg, tilt_rad, tilt_samples = max_tilt(roll, pitch)

    mocap_max = None
    mocap_pairs = 0
    mocap_max_clean = None
    mocap_removed = 0
    if has_mocap:
        mocap_max, mocap_pairs = max_speed(tm_ms, mx, my, mz)
        mocap_speeds = speed_series(tm_ms, mx, my, mz)
        mocap_speeds_clean, mocap_removed = remove_outliers_mad(mocap_speeds, args.outlier_z)
        mocap_max_clean = max(mocap_speeds_clean) if mocap_speeds_clean else None

    print(f"File: {path}")
    print(f"Rows: {rows}")
    if state_max is None:
        print("Max state speed: n/a (not enough valid samples)")
    else:
        print(f"Max state speed: {state_max:.3f} m/s (from {state_pairs} valid dt pairs)")
    if state_max_clean is None:
        print("Max state speed (outlier-removed): n/a")
    else:
        print(
            f"Max state speed (outlier-removed): {state_max_clean:.3f} m/s "
            f"(removed {state_removed}/{len(state_speeds)} samples, z>{args.outlier_z})"
        )

    if has_mocap:
        if mocap_max is None:
            print("Max mocap speed: n/a (not enough valid mocap samples)")
        else:
            print(f"Max mocap speed: {mocap_max:.3f} m/s (from {mocap_pairs} valid dt pairs)")
        if mocap_max_clean is None:
            print("Max mocap speed (outlier-removed): n/a")
        else:
            print(
                f"Max mocap speed (outlier-removed): {mocap_max_clean:.3f} m/s "
                f"(removed {mocap_removed}/{len(mocap_speeds)} samples, z>{args.outlier_z})"
            )
    else:
        print("Max mocap speed: n/a (mocap columns not present in this file)")

    if tilt_deg is None or tilt_rad is None:
        print("Max tilt: n/a (not enough valid samples)")
    else:
        print(f"Max tilt: {tilt_deg:.3f} deg ({tilt_rad:.3f} rad) from {tilt_samples} samples")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
