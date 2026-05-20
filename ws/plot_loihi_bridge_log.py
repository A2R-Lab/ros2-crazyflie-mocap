#!/usr/bin/env python3

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


def latest_log(directory: Path) -> Path | None:
    logs = sorted(
        directory.glob("loihi_bridge_log_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return logs[0] if logs else None


def value(row: dict[str, str], key: str) -> float:
    raw = row.get(key, "")
    if raw == "":
        return math.nan
    return float(raw)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def demo_segment(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    start = next((idx for idx, row in enumerate(rows) if row.get("mode") == "LOIHI_DEMO"), None)
    if start is None:
        return []
    end = len(rows)
    for idx in range(start + 1, len(rows)):
        if rows[idx].get("mode") != "LOIHI_DEMO":
            end = idx
            break
    return rows[start:end]


def series(rows: list[dict[str, str]], key: str) -> list[float]:
    return [value(row, key) for row in rows]


def time_axis(rows: list[dict[str, str]]) -> list[float]:
    t0 = value(rows[0], "monotonic_s")
    return [value(row, "monotonic_s") - t0 for row in rows]


def finite_pairs(x_values: list[float], y_values: list[float]) -> tuple[list[float], list[float]]:
    x_out = []
    y_out = []
    for x, y in zip(x_values, y_values):
        if math.isfinite(x) and math.isfinite(y):
            x_out.append(x)
            y_out.append(y)
    return x_out, y_out


def axis_limits(*values: list[float], pad_fraction: float = 0.08) -> tuple[float, float]:
    finite = [v for chunk in values for v in chunk if math.isfinite(v)]
    if not finite:
        return -1.0, 1.0
    lo = min(finite)
    hi = max(finite)
    pad = max((hi - lo) * pad_fraction, 1e-3)
    return lo - pad, hi + pad


def plot_xyz_time(ax, t, rows, prefix: str, label: str, linestyle: str = "-") -> None:
    colors = {"x": "tab:blue", "y": "tab:green", "z": "tab:red"}
    for axis in ("x", "y", "z"):
        tx, yy = finite_pairs(t, series(rows, f"{prefix}_{axis}"))
        ax.plot(tx, yy, color=colors[axis], linestyle=linestyle, linewidth=1.1, label=f"{label} {axis}")


def subtract_series(a: list[float], b: list[float]) -> list[float]:
    out = []
    for x, y in zip(a, b):
        out.append(x - y if math.isfinite(x) and math.isfinite(y) else math.nan)
    return out


def mode_spans(ax, t: list[float], rows: list[dict[str, str]]) -> None:
    colors = {
        "LOIHI_DEMO": "#fff4bf",
        "LOIHI_MANUAL": "#d7ecff",
        "LANDING": "#ffd7d7",
        "TAKEOFF": "#ddf5dd",
    }
    start_idx = 0
    current = rows[0].get("mode", "")
    for idx, row in enumerate(rows[1:], start=1):
        mode = row.get("mode", "")
        if mode == current:
            continue
        if current in colors:
            ax.axvspan(t[start_idx], t[idx - 1], color=colors[current], alpha=0.35, linewidth=0)
        start_idx = idx
        current = mode
    if current in colors:
        ax.axvspan(t[start_idx], t[-1], color=colors[current], alpha=0.35, linewidth=0)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_input = latest_log(script_dir)
    parser = argparse.ArgumentParser(description="Plot Crazyflie Loihi bridge CSV logs.")
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=default_input,
        help="Path to loihi_bridge_log_*.csv. Defaults to newest log in this directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to <input>_plot.png.",
    )
    parser.add_argument("--show", action="store_true", help="Show the plot window.")
    args = parser.parse_args()

    if args.input is None:
        raise FileNotFoundError(f"No loihi_bridge_log_*.csv found in {script_dir}")
    input_path = args.input.resolve()
    output_path = args.output or input_path.with_name(f"{input_path.stem}_plot.png")

    rows = load_rows(input_path)
    demo_rows = demo_segment(rows)
    if demo_rows:
        rows = demo_rows

    t = time_axis(rows)
    state_x = series(rows, "state_x")
    state_y = series(rows, "state_y")
    state_z = series(rows, "state_z")
    ref_x = series(rows, "ref_x")
    ref_y = series(rows, "ref_y")
    ref_z = series(rows, "ref_z")
    ref_next_x = series(rows, "ref_next_x")
    ref_next_y = series(rows, "ref_next_y")
    cmd_x = series(rows, "cmd_x")
    cmd_y = series(rows, "cmd_y")
    cmd_z = series(rows, "cmd_z")
    latency_ms = [1000.0 * v if math.isfinite(v) else math.nan for v in series(rows, "loihi_latency_s")]
    mocap_age_ms = [1000.0 * v if math.isfinite(v) else math.nan for v in series(rows, "mocap_age_s")]

    fig = plt.figure(figsize=(15, 10))
    ax_xy = fig.add_subplot(2, 3, 1)
    ax_z = fig.add_subplot(2, 3, 2)
    ax_err = fig.add_subplot(2, 3, 3)
    ax_xy_time = fig.add_subplot(2, 3, 4)
    ax_bounds = fig.add_subplot(2, 3, 5)
    ax_health = fig.add_subplot(2, 3, 6)

    sx, sy = finite_pairs(state_x, state_y)
    rx, ry = finite_pairs(ref_x, ref_y)
    cx, cy = finite_pairs(cmd_x, cmd_y)
    ax_xy.plot(rx, ry, color="black", linestyle="--", linewidth=1.2, label="reference")
    ax_xy.plot(cx, cy, color="tab:orange", linewidth=1.1, label="command")
    ax_xy.plot(sx, sy, color="tab:blue", linewidth=1.3, label="state")
    ax_xy.set_title("XY Path")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.grid(True)
    ax_xy.legend()

    mode_spans(ax_z, t, rows)
    for data, label, color, style in (
        (ref_z, "ref z", "black", "--"),
        (cmd_z, "cmd z", "tab:orange", "-"),
        (state_z, "state z", "tab:red", "-"),
    ):
        tx, yy = finite_pairs(t, data)
        ax_z.plot(tx, yy, color=color, linestyle=style, linewidth=1.1, label=label)
    ax_z.set_title("Z vs Time")
    ax_z.set_xlabel("time [s]")
    ax_z.set_ylabel("z [m]")
    ax_z.grid(True)
    ax_z.legend()

    mode_spans(ax_err, t, rows)
    for axis, color in (("x", "tab:blue"), ("y", "tab:green"), ("z", "tab:red")):
        tx, yy = finite_pairs(t, series(rows, f"e0_{axis}"))
        ax_err.plot(tx, yy, color=color, linewidth=1.1, label=f"e0 {axis}")
    ax_err.axhline(0.0, color="black", linewidth=0.8)
    ax_err.set_title("Loihi Input Position Error")
    ax_err.set_xlabel("time [s]")
    ax_err.set_ylabel("error [m]")
    ax_err.grid(True)
    ax_err.legend()

    mode_spans(ax_xy_time, t, rows)
    plot_xyz_time(ax_xy_time, t, rows, "state", "state")
    plot_xyz_time(ax_xy_time, t, rows, "ref", "ref", "--")
    world_x_min = subtract_series(ref_next_x, series(rows, "bound_ex_max"))
    world_x_max = subtract_series(ref_next_x, series(rows, "bound_ex_min"))
    world_y_min = subtract_series(ref_next_y, series(rows, "bound_ey_max"))
    world_y_max = subtract_series(ref_next_y, series(rows, "bound_ey_min"))
    for data, label, color in (
        (world_x_min, "x bound", "tab:blue"),
        (world_x_max, "x bound", "tab:blue"),
        (world_y_min, "y bound", "tab:green"),
        (world_y_max, "y bound", "tab:green"),
    ):
        tx, yy = finite_pairs(t, data)
        ax_xy_time.plot(tx, yy, color=color, linestyle=":", linewidth=1.0, label=label)
    ax_xy_time.set_title("State and Reference")
    ax_xy_time.set_xlabel("time [s]")
    ax_xy_time.set_ylabel("position [m]")
    ax_xy_time.grid(True)
    ax_xy_time.legend(ncol=2, fontsize=8)

    mode_spans(ax_bounds, t, rows)
    for key, label, color in (
        ("bound_ex_min", "ex min", "tab:blue"),
        ("bound_ex_max", "ex max", "tab:blue"),
        ("bound_ey_min", "ey min", "tab:green"),
        ("bound_ey_max", "ey max", "tab:green"),
    ):
        tx, yy = finite_pairs(t, series(rows, key))
        style = "--" if key.endswith("min") else "-"
        ax_bounds.plot(tx, yy, color=color, linestyle=style, linewidth=1.0, label=label)
    ax_bounds.axhline(0.0, color="black", linewidth=0.8)
    ax_bounds.set_title("Dynamic Error Bounds")
    ax_bounds.set_xlabel("time [s]")
    ax_bounds.set_ylabel("bound [m]")
    ax_bounds.grid(True)
    ax_bounds.legend(ncol=2, fontsize=8)

    mode_spans(ax_health, t, rows)
    tx, lat = finite_pairs(t, latency_ms)
    ax_health.plot(tx, lat, color="tab:purple", linewidth=1.1, label="Loihi latency [ms]")
    tx, age = finite_pairs(t, mocap_age_ms)
    ax_health.plot(tx, age, color="tab:brown", linewidth=1.0, label="mocap age [ms]")
    fault_t = [ti for ti, row in zip(t, rows) if row.get("fault_reason")]
    if fault_t:
        ax_health.scatter(
            fault_t,
            [0.0] * len(fault_t),
            color="tab:red",
            marker="x",
            s=20,
            label="fault row",
        )
    ax_health.set_title("Runtime Health")
    ax_health.set_xlabel("time [s]")
    ax_health.set_ylabel("ms")
    ax_health.grid(True)
    ax_health.legend(fontsize=8)

    ymin, ymax = axis_limits(state_z, ref_z, cmd_z)
    ax_z.set_ylim(ymin, ymax)

    fig.suptitle(input_path.name)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    print(f"Saved {output_path}")
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
