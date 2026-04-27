#!/usr/bin/env python3

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_OFFSET_X = -0.45
DEFAULT_OFFSET_Y = +0.55


def find_latest_log(directory: Path):
    candidates = list(directory.glob("cf_log_*.csv")) + list(directory.glob("mocap_log_*.csv"))
    logs = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


def _safe_float(value):
    if value is None:
        return float("nan")
    s = str(value).strip()
    if not s:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _int16_to_unit_float(value):
    if not math.isfinite(value):
        return float("nan")
    return value / 32767.0


def load_log_csv(csv_path: Path):
    data = {
        "time_s": [],
        "state_x": [],
        "state_y": [],
        "state_z": [],
        "controller_id": [],
        "sp_x": [],
        "sp_y": [],
        "sp_z": [],
        "u1": [],
        "u2": [],
        "u3": [],
        "u4": [],
    }
    has_setpoint = False
    has_controls = False
    has_controller = False

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])

        is_cf_log = {"time_ms", "state_x", "state_y", "state_z"}.issubset(fields)
        is_mocap_log = {"time_ms", "x", "y", "z"}.issubset(fields)
        if not is_cf_log and not is_mocap_log:
            raise ValueError(
                "Unsupported CSV format. Expected cf_log_* columns or mocap_log_* columns."
            )

        if is_cf_log:
            has_setpoint = {"setpoint_x", "setpoint_y", "setpoint_z"}.issubset(fields)
            has_controls = {"u1_16", "u2_16", "u3_16", "u4_16"}.issubset(fields)
            has_controller = "controller_id" in fields
            for row in reader:
                data["time_s"].append(_safe_float(row.get("time_ms")) / 1000.0)
                data["state_x"].append(_safe_float(row.get("state_x")))
                data["state_y"].append(_safe_float(row.get("state_y")))
                data["state_z"].append(_safe_float(row.get("state_z")))
                data["controller_id"].append(_safe_float(row.get("controller_id")) if has_controller else float("nan"))
                if has_setpoint:
                    data["sp_x"].append(_safe_float(row.get("setpoint_x")))
                    data["sp_y"].append(_safe_float(row.get("setpoint_y")))
                    data["sp_z"].append(_safe_float(row.get("setpoint_z")))
                if has_controls:
                    data["u1"].append(_int16_to_unit_float(_safe_float(row.get("u1_16"))))
                    data["u2"].append(_int16_to_unit_float(_safe_float(row.get("u2_16"))))
                    data["u3"].append(_int16_to_unit_float(_safe_float(row.get("u3_16"))))
                    data["u4"].append(_int16_to_unit_float(_safe_float(row.get("u4_16"))))
        else:
            has_setpoint = {"sp_x", "sp_y", "sp_z"}.issubset(fields)
            has_controller = "controller_id" in fields
            for row in reader:
                data["time_s"].append(_safe_float(row.get("time_ms")) / 1000.0)
                data["state_x"].append(_safe_float(row.get("x")))
                data["state_y"].append(_safe_float(row.get("y")))
                data["state_z"].append(_safe_float(row.get("z")))
                data["controller_id"].append(_safe_float(row.get("controller_id")) if has_controller else float("nan"))
                if has_setpoint:
                    data["sp_x"].append(_safe_float(row.get("sp_x")))
                    data["sp_y"].append(_safe_float(row.get("sp_y")))
                    data["sp_z"].append(_safe_float(row.get("sp_z")))

    if not data["time_s"]:
        raise ValueError(f"No data rows found in {csv_path}")

    # Normalize per-log time so different runs can be aligned/overlaid
    # regardless of wall-clock timestamp differences.
    finite_times = [t for t in data["time_s"] if math.isfinite(t)]
    if finite_times:
        t0 = finite_times[0]
        data["time_s"] = [t - t0 if math.isfinite(t) else float("nan") for t in data["time_s"]]

    return data, has_setpoint, has_controls, has_controller


def _valid(vals):
    return [v for v in vals if math.isfinite(v)]


def _xy_series(data):
    t = np.asarray(data["time_s"], dtype=float)
    x = np.asarray(data["state_x"], dtype=float)
    y = np.asarray(data["state_y"], dtype=float)

    valid = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
    t = t[valid]
    x = x[valid]
    y = y[valid]
    if t.size < 3:
        return None, None, None

    order = np.argsort(t)
    t = t[order]
    x = x[order]
    y = y[order]
    return t, x, y

def estimate_time_offset_s(reference_data, other_data):
    t_ref, x_ref, y_ref = _xy_series(reference_data)
    t_other, x_other, y_other = _xy_series(other_data)
    if t_ref is None or t_other is None:
        return 0.0

    d_ref = np.diff(t_ref)
    d_other = np.diff(t_other)
    dt_candidates = np.concatenate([d_ref[np.isfinite(d_ref) & (d_ref > 1e-9)], d_other[np.isfinite(d_other) & (d_other > 1e-9)]])
    if dt_candidates.size == 0:
        return 0.0
    dt = float(np.median(dt_candidates))
    if not math.isfinite(dt) or dt <= 0.0:
        return 0.0

    dur_ref = float(t_ref[-1] - t_ref[0])
    dur_other = float(t_other[-1] - t_other[0])
    max_shift = max(dur_ref, dur_other)
    if not math.isfinite(max_shift) or max_shift <= 0.0:
        return 0.0

    offsets = np.linspace(-max_shift, max_shift, 401)
    best_cost = float("inf")
    best_offset = 0.0

    for off in offsets:
        start = max(float(t_ref[0]), float(t_other[0] - off))
        end = min(float(t_ref[-1]), float(t_other[-1] - off))
        overlap = end - start
        if overlap < 10.0 * dt:
            continue

        n = int(overlap / dt) + 1
        if n < 20:
            continue

        t_common = np.linspace(start, end, n)
        xr = np.interp(t_common, t_ref, x_ref)
        yr = np.interp(t_common, t_ref, y_ref)
        xo = np.interp(t_common + off, t_other, x_other)
        yo = np.interp(t_common + off, t_other, y_other)

        dx = xr - xo
        dy = yr - yo
        # Remove constant XY bias so timing alignment is independent of offset in space.
        dx = dx - np.median(dx)
        dy = dy - np.median(dy)

        cost = float(np.mean(dx * dx + dy * dy))
        if not math.isfinite(cost):
            continue
        if cost < best_cost:
            best_cost = cost
            best_offset = float(off)

    return -best_offset


def _first_controller_switch_time_s(data, from_id=1, to_id=6):
    times = data.get("time_s", [])
    controllers = data.get("controller_id", [])
    if not times or not controllers:
        return None

    prev = None
    n = min(len(times), len(controllers))
    for i in range(n):
        t = times[i]
        c = controllers[i]
        if not math.isfinite(t) or not math.isfinite(c):
            continue
        ci = int(round(c))
        if prev is not None and prev == from_id and ci == to_id:
            return float(t)
        prev = ci
    return None


def main():
    script_dir = Path(__file__).resolve().parent
    latest_log = find_latest_log(script_dir)
    parser = argparse.ArgumentParser(description="Plot Crazyflie log CSV data.")
    parser.add_argument(
        "--input",
        type=Path,
        default=latest_log,
        help="Path to CSV log file (default: latest cf_log_*.csv or mocap_log_*.csv)",
    )
    parser.add_argument(
        "--input2",
        type=Path,
        default=None,
        help="Optional second CSV to overlay on the same four plots",
    )
    parser.add_argument(
        "--time-shift2",
        type=float,
        default=None,
        help=(
            "Additional manual time shift [s] for --input2 after auto alignment "
            "(controller-switch/fallback). Positive shifts it later."
        ),
    )
    parser.add_argument(
        "--offset-x",
        type=float,
        default=DEFAULT_OFFSET_X,
        help="Constant X offset [m] applied to plotted state/setpoint data",
    )
    parser.add_argument(
        "--offset-y",
        type=float,
        default=DEFAULT_OFFSET_Y,
        help="Constant Y offset [m] applied to plotted state/setpoint data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(latest_log.with_name(f"{latest_log.stem}_plot.png") if latest_log else script_dir / "cf_log_plot.png"),
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
            f"No cf_log_*.csv or mocap_log_*.csv found in {script_dir}. Provide --input explicitly."
        )

    datasets = []
    data1, has_setpoint1, has_controls1, has_controller1 = load_log_csv(args.input)
    datasets.append(
        {
            "name": args.input.name,
            "data": data1,
            "has_setpoint": has_setpoint1,
            "has_controls": has_controls1,
            "has_controller": has_controller1,
            "time_shift_s": 0.0,
        }
    )
    if args.input2 is not None:
        data2, has_setpoint2, has_controls2, has_controller2 = load_log_csv(args.input2)
        t_sw1 = _first_controller_switch_time_s(data1, from_id=1, to_id=6) if has_controller1 else None
        t_sw2 = _first_controller_switch_time_s(data2, from_id=1, to_id=6) if has_controller2 else None
        if t_sw1 is not None and t_sw2 is not None:
            auto_shift2 = t_sw1 - t_sw2
            auto_reason = "controller switch 1->6"
        else:
            auto_shift2 = estimate_time_offset_s(data1, data2)
            auto_reason = "auto fallback"

        extra_shift2 = float(args.time_shift2) if args.time_shift2 is not None else 0.0
        time_shift2 = auto_shift2 + extra_shift2
        print(
            f"Applied time shift to {args.input2.name}: {time_shift2:+.3f}s "
            f"({auto_reason}: {auto_shift2:+.3f}s, manual extra: {extra_shift2:+.3f}s)"
        )
        datasets.append(
            {
                "name": args.input2.name,
                "data": data2,
                "has_setpoint": has_setpoint2,
                "has_controls": has_controls2,
                "has_controller": has_controller2,
                "time_shift_s": time_shift2,
            }
        )

    fig = plt.figure(figsize=(12, 9))
    ax_x = fig.add_subplot(2, 2, 1)
    ax_y = fig.add_subplot(2, 2, 2)
    ax_u = fig.add_subplot(2, 2, 3)
    ax_xy = fig.add_subplot(2, 2, 4)
    x_off = float(args.offset_x)
    y_off = float(args.offset_y)

    all_xy = []
    for ds in datasets:
        d = ds["data"]
        all_xy += [v + x_off for v in _valid(d["state_x"])]
        all_xy += [v + y_off for v in _valid(d["state_y"])]
        if ds["has_setpoint"]:
            all_xy += [v + x_off for v in _valid(d["sp_x"])]
            all_xy += [v + y_off for v in _valid(d["sp_y"])]
    y_min = min(all_xy) if all_xy else -1.0
    y_max = max(all_xy) if all_xy else 1.0
    y_pad = max((y_max - y_min) * 0.05, 1e-3)
    y_min -= y_pad
    y_max += y_pad

    state_x_colors = ["tab:blue", "tab:red"]
    state_y_colors = ["tab:green", "tab:purple"]
    sp_colors = ["tab:orange", "tab:brown"]
    for i, ds in enumerate(datasets):
        d = ds["data"]
        time_s = [t + ds["time_shift_s"] for t in d["time_s"]]
        x_shifted = [v + x_off if math.isfinite(v) else float("nan") for v in d["state_x"]]
        sp_x_shifted = [v + x_off if math.isfinite(v) else float("nan") for v in d["sp_x"]]
        suffix = f" ({ds['name']})" if len(datasets) > 1 else ""
        ax_x.plot(time_s, x_shifted, color=state_x_colors[i % len(state_x_colors)], label=f"state_x{suffix}")
        if ds["has_setpoint"]:
            ax_x.plot(
                time_s,
                sp_x_shifted,
                color=sp_colors[i % len(sp_colors)],
                linestyle="--",
                label=f"setpoint_x{suffix}",
            )
    ax_x.set_title("X vs Time")
    ax_x.set_xlabel("Time [s]")
    ax_x.set_ylabel("X [m]")
    ax_x.set_ylim(y_min, y_max)
    ax_x.grid(True)
    if len(datasets) > 1 or any(ds["has_setpoint"] for ds in datasets):
        ax_x.legend()

    for i, ds in enumerate(datasets):
        d = ds["data"]
        time_s = [t + ds["time_shift_s"] for t in d["time_s"]]
        y_shifted = [v + y_off if math.isfinite(v) else float("nan") for v in d["state_y"]]
        sp_y_shifted = [v + y_off if math.isfinite(v) else float("nan") for v in d["sp_y"]]
        suffix = f" ({ds['name']})" if len(datasets) > 1 else ""
        ax_y.plot(time_s, y_shifted, color=state_y_colors[i % len(state_y_colors)], label=f"state_y{suffix}")
        if ds["has_setpoint"]:
            ax_y.plot(
                time_s,
                sp_y_shifted,
                color=sp_colors[i % len(sp_colors)],
                linestyle="--",
                label=f"setpoint_y{suffix}",
            )
    ax_y.set_title("Y vs Time")
    ax_y.set_xlabel("Time [s]")
    ax_y.set_ylabel("Y [m]")
    ax_y.set_ylim(y_min, y_max)
    ax_y.grid(True)
    if len(datasets) > 1 or any(ds["has_setpoint"] for ds in datasets):
        ax_y.legend()

    any_controls = any(ds["has_controls"] for ds in datasets)
    if any_controls:
        u_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
        for i, ds in enumerate(datasets):
            if not ds["has_controls"]:
                continue
            d = ds["data"]
            time_s = [t + ds["time_shift_s"] for t in d["time_s"]]
            suffix = f" ({ds['name']})" if len(datasets) > 1 else ""
            line_alpha = 1.0 if i == 0 else 0.75
            ax_u.plot(time_s, d["u1"], color=u_colors[0], alpha=line_alpha, linestyle="-" if i == 0 else "--", label=f"u1_16{suffix}")
            ax_u.plot(time_s, d["u2"], color=u_colors[1], alpha=line_alpha, linestyle="-" if i == 0 else "--", label=f"u2_16{suffix}")
            ax_u.plot(time_s, d["u3"], color=u_colors[2], alpha=line_alpha, linestyle="-" if i == 0 else "--", label=f"u3_16{suffix}")
            ax_u.plot(time_s, d["u4"], color=u_colors[3], alpha=line_alpha, linestyle="-" if i == 0 else "--", label=f"u4_16{suffix}")
        ax_u.legend(ncol=2)
    else:
        ax_u.text(
            0.5,
            0.5,
            "No control columns found",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax_u.transAxes,
        )
    ax_u.set_title("Controls vs Time")
    ax_u.set_xlabel("Time [s]")
    ax_u.set_ylabel("Control [0..1]")
    ax_u.set_ylim(0.0, 1.0)
    ax_u.grid(True)

    for i, ds in enumerate(datasets):
        d = ds["data"]
        suffix = f" ({ds['name']})" if len(datasets) > 1 else ""
        x_shifted = [v + x_off if math.isfinite(v) else float("nan") for v in d["state_x"]]
        y_shifted = [v + y_off if math.isfinite(v) else float("nan") for v in d["state_y"]]
        sp_x_shifted = [v + x_off if math.isfinite(v) else float("nan") for v in d["sp_x"]]
        sp_y_shifted = [v + y_off if math.isfinite(v) else float("nan") for v in d["sp_y"]]
        ax_xy.plot(
            x_shifted,
            y_shifted,
            color=state_x_colors[i % len(state_x_colors)],
            linewidth=1.5,
            label=f"state path{suffix}",
        )
        if ds["has_setpoint"]:
            ax_xy.plot(
                sp_x_shifted,
                sp_y_shifted,
                color=sp_colors[i % len(sp_colors)],
                linestyle="--",
                linewidth=1.2,
                label=f"setpoint path{suffix}",
            )
    ax_xy.set_title("XY Path")
    ax_xy.set_xlabel("X [m]")
    ax_xy.set_ylabel("Y [m]")
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.grid(True)
    ax_xy.legend()

    if len(datasets) == 1:
        fig.suptitle(f"Log: {datasets[0]['name']}", fontsize=12)
    else:
        fig.suptitle(f"Logs: {datasets[0]['name']} vs {datasets[1]['name']} (time-aligned)", fontsize=12)
    plt.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Plot saved to {args.output}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
