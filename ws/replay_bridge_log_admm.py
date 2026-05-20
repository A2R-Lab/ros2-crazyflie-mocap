#!/usr/bin/env python3
"""Replay a Crazyflie Loihi bridge CSV through host ADMM references."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np


def value(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "") or "nan")
    except ValueError:
        return math.nan


def vector_from_row(row: dict[str, str], keys: list[str]) -> np.ndarray:
    out = np.asarray([value(row, key) for key in keys], dtype=np.float64)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"row has non-finite values for {keys}")
    return out


def expand_xy_bounds_float(bounds: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    bounds = np.asarray(bounds, dtype=np.float64)
    lower = np.empty(2 * horizon, dtype=np.float64)
    upper = np.empty(2 * horizon, dtype=np.float64)
    lower[0::2] = bounds[0]
    lower[1::2] = bounds[2]
    upper[0::2] = bounds[1]
    upper[1::2] = bounds[3]
    return lower, upper


def rescale_problem_rho(problem_data: dict, rho_scale: float | None, rho: float | None) -> dict:
    if rho_scale is None and rho is None:
        return problem_data
    out = dict(problem_data)
    old_rho_vect = np.asarray(problem_data["rho_vect"], dtype=np.float64)
    if rho is not None:
        new_rho_vect = np.full_like(old_rho_vect, float(rho), dtype=np.float64)
    else:
        new_rho_vect = old_rho_vect * float(rho_scale)
    a = np.asarray(problem_data["A"], dtype=np.float64)
    p = np.asarray(problem_data["P"], dtype=np.float64)
    atrho = a.T @ np.diag(new_rho_vect)
    kkt = p + atrho @ a
    l_factor = np.linalg.cholesky(kkt)
    out["ATrho"] = atrho
    out["KKT"] = kkt
    out["L"] = l_factor
    out["LT"] = l_factor.T
    out["rho_vect"] = new_rho_vect
    out["RHO"] = float(new_rho_vect[0]) if new_rho_vect.size else 0.0
    return out


def selected_violation(selected: np.ndarray, bounds: np.ndarray) -> float:
    return float(
        max(
            bounds[0] - selected[0],
            selected[0] - bounds[1],
            bounds[2] - selected[1],
            selected[1] - bounds[3],
            0.0,
        )
    )


def run_float_step(
    problem_data: dict,
    z_state: np.ndarray,
    y_state: np.ndarray,
    current_state: np.ndarray,
    bounds: np.ndarray,
    iterations: int,
    equality_rows: int,
    input_inequality_rows: int,
    xy_inequality_rows: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(problem_data["A"], dtype=np.float64)
    atrho = np.asarray(problem_data["ATrho"], dtype=np.float64)
    l_factor = np.asarray(problem_data["L"], dtype=np.float64)
    lower, upper = problem_data["_float_lower"].copy(), problem_data["_float_upper"].copy()
    xy_lower, xy_upper = expand_xy_bounds_float(bounds, horizon)
    xy_start = equality_rows + input_inequality_rows
    xy_end = xy_start + xy_inequality_rows
    lower[xy_start:xy_end] = xy_lower
    upper[xy_start:xy_end] = xy_upper

    z = np.asarray(z_state, dtype=np.float64).copy()
    y = np.asarray(y_state, dtype=np.float64).copy()
    z[: current_state.shape[0]] = current_state
    x = np.zeros(a.shape[1], dtype=np.float64)
    for _ in range(iterations):
        rhs = atrho @ (z - y)
        w = np.linalg.solve(l_factor, rhs)
        x = np.linalg.solve(l_factor.T, w)
        v = a @ x + y
        z_next = np.empty_like(v)
        z_next[: current_state.shape[0]] = current_state
        z_next[current_state.shape[0] : equality_rows] = 0.0
        z_next[equality_rows:] = np.minimum(
            np.maximum(v[equality_rows:], lower[equality_rows:]),
            upper[equality_rows:],
        )
        y = v - z_next
        z = z_next
    return x, z, y


def summarize(values: list[float]) -> str:
    finite = sorted(v for v in values if math.isfinite(v))
    if not finite:
        return "n/a"
    return (
        f"mean={sum(finite) / len(finite):.6f} "
        f"p95={finite[int(0.95 * (len(finite) - 1))]:.6f} "
        f"max={max(finite):.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_csv", type=Path)
    parser.add_argument("--admm-nxcore", type=Path, default=Path("/home/agrillo/neuromorphic/admm_nxcore"))
    parser.add_argument("--mode", default="LOIHI_DEMO")
    parser.add_argument("--iterations", type=int, default=45)
    parser.add_argument("--selected-position-index", type=int, default=3)
    parser.add_argument("--rho-scale", type=float)
    parser.add_argument("--rho", type=float)
    parser.add_argument(
        "--reset-y-each-step",
        action="store_true",
        help="Reset the ADMM dual y state to zero before each logged MPC request.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(args.admm_nxcore.expanduser().resolve()))
    from admm_mpc.core import choose_vector_exp, dequantize_vector, quantize_vector
    from admm_mpc.direct import build_direct_solve_pipeline, infer_inequality_split, run_direct_repeated_host_quantized
    from admm_mpc.pipeline import build_repeated_mpc_pipeline
    from script.header_generator import build_default_bounds, get_solver_problem_data

    rows = [row for row in csv.DictReader(args.log_csv.open()) if row.get("mode") == args.mode]
    if not rows:
        raise ValueError(f"No rows with mode={args.mode!r} in {args.log_csv}")

    problem_data = rescale_problem_rho(
        get_solver_problem_data(),
        rho_scale=args.rho_scale,
        rho=args.rho,
    )
    state_dim = int(problem_data["n"])
    control_dim = int(problem_data["m"])
    equality_rows = int(problem_data["A_eq"].shape[0])
    input_ineq_rows, xy_ineq_rows, horizon = infer_inequality_split(
        problem_data["A"].shape[0],
        equality_rows,
        control_dim,
    )
    selected_index = int(args.selected_position_index)
    if selected_index < 0:
        selected_index = horizon
    selected_slice = slice(
        selected_index * (state_dim + control_dim),
        selected_index * (state_dim + control_dim) + 3,
    )

    vector_values = np.concatenate(
        [
            problem_data["u_min"],
            problem_data["u_max"],
            np.asarray([-4.4, 1.0, -1.0], dtype=np.float64),
        ]
    )
    vector_exp = choose_vector_exp(vector_values)
    lower, upper = build_default_bounds(np.zeros(state_dim, dtype=np.float64))
    problem_data["_float_lower"] = lower
    problem_data["_float_upper"] = upper
    repeated_pipeline = build_repeated_mpc_pipeline(problem_data, lower, upper, vector_exp)
    direct_pipeline = build_direct_solve_pipeline(
        problem_data,
        repeated_pipeline,
        selected_position_index=selected_index,
    )

    z_float = np.zeros(problem_data["A"].shape[0], dtype=np.float64)
    y_float = np.zeros(problem_data["A"].shape[0], dtype=np.float64)
    z_quant = np.zeros(problem_data["A"].shape[0], dtype=np.int64)
    y_quant = np.zeros(problem_data["A"].shape[0], dtype=np.int64)

    output_path = args.output or args.log_csv.with_name(f"{args.log_csv.stem}_admm_replay.csv")
    fieldnames = [
        "row",
        "t_s",
        "request_epoch",
        "loihi_x",
        "loihi_y",
        "loihi_z",
        "float_x",
        "float_y",
        "float_z",
        "quant_x",
        "quant_y",
        "quant_z",
        "bound_ex_min",
        "bound_ex_max",
        "bound_ey_min",
        "bound_ey_max",
        "loihi_violation",
        "float_violation",
        "quant_violation",
        "loihi_float_err",
        "loihi_quant_err",
        "float_quant_err",
    ]
    t0 = value(rows[0], "monotonic_s")
    state_keys = [
        "e0_x",
        "e0_y",
        "e0_z",
        "e0_phi_x",
        "e0_phi_y",
        "e0_phi_z",
        "e0_vx",
        "e0_vy",
        "e0_vz",
        "e0_omega_x",
        "e0_omega_y",
        "e0_omega_z",
    ]
    bounds_keys = ["bound_ex_min", "bound_ex_max", "bound_ey_min", "bound_ey_max"]
    metrics = {
        "loihi_violation": [],
        "float_violation": [],
        "quant_violation": [],
        "loihi_float_err": [],
        "loihi_quant_err": [],
        "float_quant_err": [],
    }
    crossings: dict[str, tuple[int, float, float] | None] = {
        "loihi_violation_0.1": None,
        "quant_violation_0.1": None,
        "loihi_quant_err_0.1": None,
    }

    with output_path.open("w", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()
        replayed = 0
        skipped = 0
        for idx, row in enumerate(rows):
            try:
                state = vector_from_row(row, state_keys)
                bounds = vector_from_row(row, bounds_keys)
                loihi = vector_from_row(row, ["selected_ep_x", "selected_ep_y", "selected_ep_z"])
            except ValueError:
                skipped += 1
                continue
            if args.reset_y_each_step:
                y_float.fill(0.0)
                y_quant.fill(0)

            x_float, z_float, y_float = run_float_step(
                problem_data,
                z_float,
                y_float,
                state,
                bounds,
                args.iterations,
                equality_rows,
                input_ineq_rows,
                xy_ineq_rows,
                horizon,
            )
            state_int = quantize_vector(state, vector_exp)
            bounds_int = quantize_vector(bounds, vector_exp)
            quant_result = run_direct_repeated_host_quantized(
                direct_pipeline,
                state_int,
                y_quant,
                z_quant,
                args.iterations,
                "output-only",
                xy_bounds_int=bounds_int,
            )
            z_quant = quant_result["z"]
            y_quant = quant_result["y"]

            selected_float = np.asarray(x_float[selected_slice], dtype=np.float64)
            selected_quant = dequantize_vector(quant_result["selected_xyz"], vector_exp)

            row_metrics = {
                "loihi_violation": selected_violation(loihi, bounds),
                "float_violation": selected_violation(selected_float, bounds),
                "quant_violation": selected_violation(selected_quant, bounds),
                "loihi_float_err": float(np.linalg.norm(loihi - selected_float)),
                "loihi_quant_err": float(np.linalg.norm(loihi - selected_quant)),
                "float_quant_err": float(np.linalg.norm(selected_float - selected_quant)),
            }
            for key, metric in row_metrics.items():
                metrics[key].append(metric)
            t_s = value(row, "monotonic_s") - t0
            for key, threshold in (
                ("loihi_violation_0.1", row_metrics["loihi_violation"]),
                ("quant_violation_0.1", row_metrics["quant_violation"]),
                ("loihi_quant_err_0.1", row_metrics["loihi_quant_err"]),
            ):
                if crossings[key] is None and threshold > 0.1:
                    crossings[key] = (idx, t_s, threshold)

            writer.writerow(
                {
                    "row": idx,
                    "t_s": f"{t_s:.6f}",
                    "request_epoch": row.get("request_epoch", ""),
                    "loihi_x": f"{loihi[0]:.9f}",
                    "loihi_y": f"{loihi[1]:.9f}",
                    "loihi_z": f"{loihi[2]:.9f}",
                    "float_x": f"{selected_float[0]:.9f}",
                    "float_y": f"{selected_float[1]:.9f}",
                    "float_z": f"{selected_float[2]:.9f}",
                    "quant_x": f"{selected_quant[0]:.9f}",
                    "quant_y": f"{selected_quant[1]:.9f}",
                    "quant_z": f"{selected_quant[2]:.9f}",
                    "bound_ex_min": f"{bounds[0]:.9f}",
                    "bound_ex_max": f"{bounds[1]:.9f}",
                    "bound_ey_min": f"{bounds[2]:.9f}",
                    "bound_ey_max": f"{bounds[3]:.9f}",
                    **{key: f"{val:.9f}" for key, val in row_metrics.items()},
                }
            )
            replayed += 1

    print(f"rows_replayed {replayed}")
    print(f"rows_skipped {skipped}")
    print(f"output {output_path}")
    print(f"vector_exp {vector_exp}")
    print(f"rho {float(problem_data['RHO'])}")
    print(f"reset_y_each_step {int(args.reset_y_each_step)}")
    print(f"horizon {horizon}")
    print(f"selected_position_index {selected_index}")
    for key, values in metrics.items():
        print(f"{key} {summarize(values)}")
    for key, crossing in crossings.items():
        print(f"{key} {crossing}")


if __name__ == "__main__":
    main()
