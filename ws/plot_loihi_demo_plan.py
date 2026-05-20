#!/usr/bin/env python3

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from crazyflie_bridge.crazyflie_bridge_node import (
    BridgeConfig,
    CircleReferenceGenerator,
    DEFAULT_BOX_AXIS_HOLD_PERIODS,
    DEFAULT_BOX_AXIS_OPEN_PERIODS,
    DEFAULT_BOX_AXIS_SHRINK_PERIODS,
    DEFAULT_BOX_BETWEEN_AXES_OPEN_PERIODS,
    DEFAULT_BOX_FINAL_OPEN_PERIODS,
    DEFAULT_BOX_INITIAL_OPEN_PERIODS,
    DEFAULT_BOX_NARROW_HALF_EXTENT_M,
    DEFAULT_BOX_OPEN_HALF_EXTENT_M,
    DEFAULT_CIRCLE_CENTER_M,
    DEFAULT_CIRCLE_OMEGA_RAD_S,
    DEFAULT_CIRCLE_RADIUS_M,
    DynamicBoundsShifter,
)


def demo_cycle_periods(config: BridgeConfig) -> float:
    axis_stage = (
        max(1.0e-6, float(config.box_axis_shrink_periods))
        + max(0.0, float(config.box_axis_hold_periods))
        + max(1.0e-6, float(config.box_axis_open_periods))
    )
    return (
        max(0.0, float(config.box_initial_open_periods))
        + axis_stage
        + max(0.0, float(config.box_between_axes_open_periods))
        + axis_stage
        + max(0.0, float(config.box_final_open_periods))
    )


def stage_name(phase: float, config: BridgeConfig) -> str:
    initial_open = max(0.0, float(config.box_initial_open_periods))
    axis_stage = (
        max(1.0e-6, float(config.box_axis_shrink_periods))
        + max(0.0, float(config.box_axis_hold_periods))
        + max(1.0e-6, float(config.box_axis_open_periods))
    )
    between_axes = max(0.0, float(config.box_between_axes_open_periods))
    x_end = initial_open + axis_stage
    y_start = x_end + between_axes
    y_end = y_start + axis_stage
    if phase < initial_open:
        return "open circle"
    if phase < x_end:
        return "x shrink/hold/open"
    if phase < y_start:
        return "open transition"
    if phase < y_end:
        return "y shrink/hold/open"
    return "open circle"


def make_bridge_config(args: argparse.Namespace) -> BridgeConfig:
    return BridgeConfig(
        uri="",
        use_mocap=True,
        mocap_mode="position_only",
        axis_mapping=[0, 1, 2],
        axis_sign=[1.0, 1.0, 1.0],
        yaw_offset_rad=0.0,
        control_period_s=0.01,
        extpos_period_s=0.01,
        state_log_period_ms=10,
        max_state_age_s=0.1,
        max_mocap_age_s=0.25,
        position_commands_enabled=False,
        auto_takeoff=False,
        auto_start_demo=False,
        arm_on_connect=False,
        takeoff_height_m=0.5,
        takeoff_rate_mps=0.25,
        land_rate_mps=0.25,
        z_min_m=0.1,
        z_max_m=1.2,
        max_command_step_m=0.05,
        fault_land_count=40,
        fixed_yaw_deg=0.0,
        reference_yaw_mode="fixed",
        circle_radius_m=float(args.circle_radius_m),
        circle_omega_rad_s=float(args.circle_omega_rad_s),
        circle_center_m=np.asarray(args.circle_center, dtype=np.float64),
        box_l0_m=float(args.box_open_half_extent_m),
        box_min_half_extent_m=float(args.box_narrow_half_extent_m),
        box_initial_open_periods=float(args.box_initial_open_periods),
        box_axis_shrink_periods=float(args.box_axis_shrink_periods),
        box_axis_hold_periods=float(args.box_axis_hold_periods),
        box_axis_open_periods=float(args.box_axis_open_periods),
        box_between_axes_open_periods=float(args.box_between_axes_open_periods),
        box_final_open_periods=float(args.box_final_open_periods),
        manual_step_m=0.1,
        loihi_backend="disabled",
        mock_error_gain=0.0,
        loihi_admm_iterations=45,
        loihi_parallel_components=3,
        loihi_parallel_quant_bins=750,
        loihi_vector_max_abs_int=-1,
        loihi_max_control_ticks=60000,
        loihi_selected_position_index=-1,
        loihi_match_timeout_s=0.05,
        max_loihi_latency_s=0.05,
        loihi_ethernet_output_buffer_steps=4096,
        admm_nxcore_path="",
        log_file="",
        print_every=100,
    )


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Preview the Loihi circular demo and staged box bounds.")
    parser.add_argument("--circle-radius-m", type=float, default=DEFAULT_CIRCLE_RADIUS_M)
    parser.add_argument("--circle-omega-rad-s", type=float, default=DEFAULT_CIRCLE_OMEGA_RAD_S)
    parser.add_argument("--circle-center", type=float, nargs=3, default=DEFAULT_CIRCLE_CENTER_M)
    parser.add_argument("--box-open-half-extent-m", type=float, default=DEFAULT_BOX_OPEN_HALF_EXTENT_M)
    parser.add_argument("--box-narrow-half-extent-m", type=float, default=DEFAULT_BOX_NARROW_HALF_EXTENT_M)
    parser.add_argument("--box-initial-open-periods", type=float, default=DEFAULT_BOX_INITIAL_OPEN_PERIODS)
    parser.add_argument("--box-axis-shrink-periods", type=float, default=DEFAULT_BOX_AXIS_SHRINK_PERIODS)
    parser.add_argument("--box-axis-hold-periods", type=float, default=DEFAULT_BOX_AXIS_HOLD_PERIODS)
    parser.add_argument("--box-axis-open-periods", type=float, default=DEFAULT_BOX_AXIS_OPEN_PERIODS)
    parser.add_argument(
        "--box-between-axes-open-periods",
        type=float,
        default=DEFAULT_BOX_BETWEEN_AXES_OPEN_PERIODS,
    )
    parser.add_argument("--box-final-open-periods", type=float, default=DEFAULT_BOX_FINAL_OPEN_PERIODS)
    parser.add_argument("--samples", type=int, default=800)
    parser.add_argument("--cycles", type=float, default=1.0, help="Number of staged demo cycles to preview.")
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "loihi_demo_plan.png",
        help="Output PNG path.",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    config = make_bridge_config(args)
    reference_generator = CircleReferenceGenerator(config)
    bounds_shifter = DynamicBoundsShifter(config)
    omega = max(abs(float(config.circle_omega_rad_s)), 1.0e-6)
    circle_period_s = 2.0 * math.pi / omega
    demo_period_s = demo_cycle_periods(config) * circle_period_s
    duration_s = demo_period_s * float(args.cycles)
    n = max(int(args.samples), 16)
    times = [duration_s * i / (n - 1) for i in range(n)]

    center_x, center_y, center_z = [float(v) for v in config.circle_center_m]
    radius = float(args.circle_radius_m)
    open_extent = float(args.box_open_half_extent_m)

    xs = []
    ys = []
    lxs = []
    lys = []
    phases = []
    for t_s in times:
        reference = reference_generator.reference_at(t_s, yaw_hold_rad=0.0)
        bounds = bounds_shifter.bounds_for(reference)
        lx = 0.5 * (float(bounds[1]) - float(bounds[0]))
        ly = 0.5 * (float(bounds[3]) - float(bounds[2]))
        stage_phase = (t_s % demo_period_s) / circle_period_s
        xs.append(float(reference.position_m[0]))
        ys.append(float(reference.position_m[1]))
        lxs.append(lx)
        lys.append(ly)
        phases.append(stage_phase)

    fig = plt.figure(figsize=(14, 8))
    ax_xy = fig.add_subplot(1, 2, 1)
    ax_time = fig.add_subplot(2, 2, 2)
    ax_violation = fig.add_subplot(2, 2, 4)

    ax_xy.plot(xs, ys, color="black", linewidth=1.8, label="reference circle")
    ax_xy.scatter([xs[0]], [ys[0]], color="tab:green", s=45, label="start")
    ax_xy.scatter([xs[-1]], [ys[-1]], color="tab:red", s=35, label="end")

    colors = {
        "open circle": "0.65",
        "open transition": "0.75",
        "x shrink/hold/open": "tab:blue",
        "y shrink/hold/open": "tab:green",
    }
    labels_seen = set()
    stride = max(n // 80, 1)
    for i in range(0, n, stride):
        label = stage_name(phases[i], config)
        rect_label = label if label not in labels_seen else None
        labels_seen.add(label)
        ax_xy.plot(
            [-lxs[i], lxs[i], lxs[i], -lxs[i], -lxs[i]],
            [-lys[i], -lys[i], lys[i], lys[i], -lys[i]],
            color=colors[label],
            alpha=0.28,
            linewidth=0.8,
            label=rect_label,
        )

    limit = max(open_extent, radius + max(abs(center_x), abs(center_y))) * 1.12
    ax_xy.set_xlim(-limit, limit)
    ax_xy.set_ylim(-limit, limit)
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.set_title("Reference and World-Frame Box")
    ax_xy.grid(True)
    ax_xy.legend(fontsize=8)

    ax_time.plot(times, lxs, color="tab:blue", label="Lx")
    ax_time.plot(times, lys, color="tab:green", label="Ly")
    ax_time.axhline(radius, color="black", linestyle="--", linewidth=1.0, label="circle radius")
    for k in range(1, int(math.floor(demo_cycle_periods(config) * args.cycles)) + 1):
        ax_time.axvline(k * circle_period_s, color="0.8", linewidth=0.8)
    ax_time.set_title("Box Half-Extents")
    ax_time.set_xlabel("time [s]")
    ax_time.set_ylabel("half-extent [m]")
    ax_time.grid(True)
    ax_time.legend(fontsize=8)

    x_margin = [lx - abs(x) for lx, x in zip(lxs, xs)]
    y_margin = [ly - abs(y) for ly, y in zip(lys, ys)]
    ax_violation.plot(times, x_margin, color="tab:blue", label="x margin")
    ax_violation.plot(times, y_margin, color="tab:green", label="y margin")
    ax_violation.axhline(0.0, color="black", linewidth=1.0)
    for k in range(1, int(math.floor(demo_cycle_periods(config) * args.cycles)) + 1):
        ax_violation.axvline(k * circle_period_s, color="0.8", linewidth=0.8)
    ax_violation.set_title("Reference Margin to Box")
    ax_violation.set_xlabel("time [s]")
    ax_violation.set_ylabel("margin [m]")
    ax_violation.grid(True)
    ax_violation.legend(fontsize=8)

    fig.suptitle(
        "Loihi Demo Preview: "
        f"R={radius:.2f} m, omega={omega:.2f} rad/s, "
        f"Lopen={open_extent:.2f} m, Lnarrow={config.box_min_half_extent_m:.2f} m, "
        f"z={center_z:.2f} m"
    )
    fig.tight_layout()
    fig.savefig(args.output, dpi=160)
    print(f"Saved {args.output}")
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
