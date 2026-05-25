#!/usr/bin/env python3

import atexit
import csv
import logging
import math
import select
import sys
import termios
import threading
import time
import traceback
import tty
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper
from cflib.utils.encoding import decompress_quaternion

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node


URI = uri_helper.uri_from_env(default="radio://0/80/2M/E7E7E7E7E7")
DEFAULT_CIRCLE_RADIUS_M = 0.25
DEFAULT_CIRCLE_OMEGA_RAD_S = 1.40
DEFAULT_CIRCLE_CENTER_M = [0.0, 0.0, 0.5]
DEFAULT_BOX_OPEN_HALF_EXTENT_M = 0.7
DEFAULT_BOX_NARROW_HALF_EXTENT_M = 0.08
DEFAULT_BOX_INITIAL_OPEN_PERIODS = 1.5
DEFAULT_BOX_AXIS_SHRINK_PERIODS = 0.7
DEFAULT_BOX_AXIS_HOLD_PERIODS = 2.0
DEFAULT_BOX_AXIS_OPEN_PERIODS = 1.0
DEFAULT_BOX_BETWEEN_AXES_OPEN_PERIODS = 0.75
DEFAULT_BOX_FINAL_OPEN_PERIODS = 1.0

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def quaternion_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_to_quaternion(yaw_rad: float) -> Tuple[float, float, float, float]:
    half_yaw = 0.5 * yaw_rad
    return 0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)


def apply_yaw_offset_to_quaternion(
    qx: float,
    qy: float,
    qz: float,
    qw: float,
    yaw_offset_rad: float,
) -> Tuple[float, float, float, float]:
    half_angle = 0.5 * yaw_offset_rad
    offset_qw = math.cos(half_angle)
    offset_qz = math.sin(half_angle)
    return (
        offset_qw * qx + offset_qz * qy,
        offset_qw * qy - offset_qz * qx,
        offset_qw * qz + offset_qz * qw,
        offset_qw * qw - offset_qz * qz,
    )


def normalize_quaternion(q: Sequence[float]) -> np.ndarray:
    q_arr = np.asarray(q, dtype=np.float64)
    norm = float(np.linalg.norm(q_arr))
    if norm <= 0.0:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q_arr / norm


def quat_mul(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.asarray(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float64,
    )


def rodrigues_attitude_error(
    q_state_xyzw: Sequence[float],
    desired_yaw_rad: float,
) -> np.ndarray:
    """Compute q_desired_yaw_conj * q_state as Rodrigues parameters."""

    q_state = normalize_quaternion(q_state_xyzw)
    q_desired_yaw_conj = np.asarray(
        [
            0.0,
            0.0,
            -math.sin(0.5 * desired_yaw_rad),
            math.cos(0.5 * desired_yaw_rad),
        ],
        dtype=np.float64,
    )
    q_error = normalize_quaternion(quat_mul(q_desired_yaw_conj, q_state))
    if abs(float(q_error[3])) < 1.0e-6:
        return np.zeros(3, dtype=np.float64)
    return q_error[:3] / q_error[3]


def default_admm_nxcore_path() -> str:
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[5]
    return str(repo_root / "admm_nxcore")


def finite_or_none(values: Optional[Sequence[float]]) -> Optional[np.ndarray]:
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float64)
    if np.all(np.isfinite(arr)):
        return arr
    return None


@dataclass
class FirmwareState:
    position_m: np.ndarray
    velocity_mps: np.ndarray
    quaternion_xyzw: np.ndarray
    gyro_rad_s: np.ndarray
    yaw_rad: float
    received_s: float
    raw: Dict[str, float]


@dataclass
class MocapSample:
    position_m: np.ndarray
    quaternion_xyzw: np.ndarray
    mode: str
    received_s: float


@dataclass
class ReferencePoint:
    position_m: np.ndarray
    velocity_mps: np.ndarray
    yaw_rad: float
    t_s: float


@dataclass
class LoihiRequest:
    epoch: int
    state_error: np.ndarray
    bounds_error_xy: np.ndarray
    reference_now: ReferencePoint
    reference_next: ReferencePoint
    sent_s: float


@dataclass
class LoihiResult:
    output_epoch: int
    u0: np.ndarray
    selected_error_position_m: np.ndarray
    latency_s: float
    raw_output: Optional[np.ndarray] = None


@dataclass
class Command:
    position_m: np.ndarray
    yaw_rad: float


@dataclass
class BridgeConfig:
    uri: str
    use_mocap: bool
    mocap_mode: str
    axis_mapping: List[int]
    axis_sign: List[float]
    yaw_offset_rad: float
    control_period_s: float
    extpos_period_s: float
    state_log_period_ms: int
    max_state_age_s: float
    max_mocap_age_s: float
    position_commands_enabled: bool
    auto_takeoff: bool
    auto_start_demo: bool
    arm_on_connect: bool
    takeoff_height_m: float
    takeoff_rate_mps: float
    land_rate_mps: float
    z_min_m: float
    z_max_m: float
    max_command_step_m: float
    fault_land_count: int
    fixed_yaw_deg: float
    reference_yaw_mode: str
    circle_radius_m: float
    circle_omega_rad_s: float
    circle_center_m: np.ndarray
    box_l0_m: float
    box_min_half_extent_m: float
    box_initial_open_periods: float
    box_axis_shrink_periods: float
    box_axis_hold_periods: float
    box_axis_open_periods: float
    box_between_axes_open_periods: float
    box_final_open_periods: float
    box_schedule_mode: str
    box_fixed_duration_s: float
    manual_step_m: float
    loihi_backend: str
    mock_error_gain: float
    loihi_admm_iterations: int
    loihi_parallel_components: int
    loihi_parallel_quant_bins: int
    loihi_vector_max_abs_int: int
    loihi_max_control_ticks: int
    loihi_selected_position_index: int
    loihi_xy_dynamic_bounds_start_index: int
    loihi_no_warm_start: bool
    loihi_match_timeout_s: float
    max_loihi_latency_s: float
    loihi_ethernet_output_buffer_steps: int
    admm_nxcore_path: str
    log_file: str
    print_every: int


class StateEstimateZLogAdapter:
    def __init__(self, period_ms: int):
        self.period_ms = int(period_ms)
        self._lock = threading.Lock()
        self._latest: Optional[FirmwareState] = None
        self._log_conf: Optional[LogConfig] = None

    def start(self, cf) -> None:
        log_conf = LogConfig(name="StateEstimateZ", period_in_ms=self.period_ms)
        for name in (
            "stateEstimateZ.x",
            "stateEstimateZ.y",
            "stateEstimateZ.z",
            "stateEstimateZ.vx",
            "stateEstimateZ.vy",
            "stateEstimateZ.vz",
            "stateEstimateZ.rateRoll",
            "stateEstimateZ.ratePitch",
            "stateEstimateZ.rateYaw",
        ):
            log_conf.add_variable(name, "int16_t")
        log_conf.add_variable("stateEstimateZ.quat", "uint32_t")

        cf.log.add_config(log_conf)
        log_conf.data_received_cb.add_callback(self._log_callback)
        log_conf.start()
        self._log_conf = log_conf
        LOGGER.info("Started stateEstimateZ firmware log at %d ms", self.period_ms)

    def latest(self) -> Optional[FirmwareState]:
        with self._lock:
            return self._latest

    def latest_age_s(self, now_s: float) -> Optional[float]:
        sample = self.latest()
        if sample is None:
            return None
        return now_s - sample.received_s

    def _log_callback(self, timestamp, data, logconf) -> None:
        try:
            quat = normalize_quaternion(decompress_quaternion(int(data["stateEstimateZ.quat"])))
            position_m = np.asarray(
                [
                    float(data["stateEstimateZ.x"]) / 1000.0,
                    float(data["stateEstimateZ.y"]) / 1000.0,
                    float(data["stateEstimateZ.z"]) / 1000.0,
                ],
                dtype=np.float64,
            )
            velocity_mps = np.asarray(
                [
                    float(data["stateEstimateZ.vx"]) / 1000.0,
                    float(data["stateEstimateZ.vy"]) / 1000.0,
                    float(data["stateEstimateZ.vz"]) / 1000.0,
                ],
                dtype=np.float64,
            )
            gyro_rad_s = np.asarray(
                [
                    float(data["stateEstimateZ.rateRoll"]) / 1000.0,
                    -float(data["stateEstimateZ.ratePitch"]) / 1000.0,
                    float(data["stateEstimateZ.rateYaw"]) / 1000.0,
                ],
                dtype=np.float64,
            )
            sample = FirmwareState(
                position_m=position_m,
                velocity_mps=velocity_mps,
                quaternion_xyzw=quat,
                gyro_rad_s=gyro_rad_s,
                yaw_rad=quaternion_to_yaw(quat[0], quat[1], quat[2], quat[3]),
                received_s=time.monotonic(),
                raw={key: float(value) for key, value in data.items()},
            )
            with self._lock:
                self._latest = sample
        except Exception as exc:
            LOGGER.warning("Could not parse stateEstimateZ log sample: %r", exc)


class MocapAdapter:
    def __init__(
        self,
        axis_mapping: Sequence[int],
        axis_sign: Sequence[float],
        yaw_offset_rad: float,
        mocap_mode: str,
    ):
        if len(axis_mapping) != 3:
            raise ValueError("axis_mapping must contain three entries.")
        if len(axis_sign) != 3:
            raise ValueError("axis_sign must contain three entries.")
        self.axis_mapping = [int(value) for value in axis_mapping]
        self.axis_sign = [float(value) for value in axis_sign]
        self.yaw_offset_rad = float(yaw_offset_rad)
        self.mocap_mode = str(mocap_mode)
        self._lock = threading.Lock()
        self._latest: Optional[MocapSample] = None

    def update(self, msg: PoseStamped) -> None:
        pos_raw = [
            float(msg.pose.position.x),
            float(msg.pose.position.y),
            float(msg.pose.position.z),
        ]
        position_m = np.asarray(
            [
                self.axis_sign[0] * pos_raw[self.axis_mapping[0]],
                self.axis_sign[1] * pos_raw[self.axis_mapping[1]],
                self.axis_sign[2] * pos_raw[self.axis_mapping[2]],
            ],
            dtype=np.float64,
        )

        if self.mocap_mode == "position_only":
            quaternion_xyzw = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        else:
            q_raw = [
                float(msg.pose.orientation.x),
                float(msg.pose.orientation.y),
                float(msg.pose.orientation.z),
            ]
            qx = self.axis_sign[0] * q_raw[self.axis_mapping[0]]
            qy = self.axis_sign[1] * q_raw[self.axis_mapping[1]]
            qz = self.axis_sign[2] * q_raw[self.axis_mapping[2]]
            qw = float(msg.pose.orientation.w)
            if self.axis_sign[0] * self.axis_sign[1] * self.axis_sign[2] < 0.0:
                qw = -qw
            if self.yaw_offset_rad != 0.0:
                qx, qy, qz, qw = apply_yaw_offset_to_quaternion(
                    qx,
                    qy,
                    qz,
                    qw,
                    self.yaw_offset_rad,
                )
            quaternion_xyzw = normalize_quaternion([qx, qy, qz, qw])

        sample = MocapSample(
            position_m=position_m,
            quaternion_xyzw=quaternion_xyzw,
            mode=self.mocap_mode,
            received_s=time.monotonic(),
        )
        with self._lock:
            self._latest = sample

    def latest(self) -> Optional[MocapSample]:
        with self._lock:
            return self._latest

    def latest_age_s(self, now_s: float) -> Optional[float]:
        sample = self.latest()
        if sample is None:
            return None
        return now_s - sample.received_s

    def send_external_pose(self, cf, firmware_state: Optional[FirmwareState]) -> bool:
        sample = self.latest()
        if sample is None:
            return False
        x, y, z = sample.position_m.tolist()
        if sample.mode == "position_only":
            cf.extpos.send_extpos(x, y, z)
            return True
        if sample.mode == "position_and_yaw":
            yaw_rad = quaternion_to_yaw(*sample.quaternion_xyzw)
            qx, qy, qz, qw = yaw_to_quaternion(yaw_rad)
            cf.extpos.send_extpose(x, y, z, qx, qy, qz, qw)
            return True
        cf.extpos.send_extpose(x, y, z, *sample.quaternion_xyzw.tolist())
        return True


class CircleReferenceGenerator:
    def __init__(self, config: BridgeConfig):
        self.config = config
        if self.config.box_schedule_mode not in ("staged", "fixed"):
            raise ValueError("box_schedule_mode must be 'staged' or 'fixed'")

    def reference_at(self, elapsed_s: float, yaw_hold_rad: float) -> ReferencePoint:
        t_s = float(elapsed_s)
        radius = float(self.config.circle_radius_m)
        omega = float(self.config.circle_omega_rad_s)
        phase = omega * t_s
        center = self.config.circle_center_m
        position = np.asarray(
            [
                center[0] + radius * math.cos(phase),
                center[1] + radius * math.sin(phase),
                center[2],
            ],
            dtype=np.float64,
        )
        velocity = np.asarray(
            [
                -radius * omega * math.sin(phase),
                radius * omega * math.cos(phase),
                0.0,
            ],
            dtype=np.float64,
        )
        yaw_mode = self.config.reference_yaw_mode
        if yaw_mode == "current":
            yaw_rad = yaw_hold_rad
        elif yaw_mode == "tangent":
            yaw_rad = math.atan2(velocity[1], velocity[0]) if radius > 0.0 else yaw_hold_rad
        else:
            yaw_rad = math.radians(float(self.config.fixed_yaw_deg))
        return ReferencePoint(position_m=position, velocity_mps=velocity, yaw_rad=yaw_rad, t_s=t_s)


class DynamicBoundsShifter:
    def __init__(self, config: BridgeConfig):
        self.config = config

    def cycle_period_s(self) -> float:
        if self.config.box_schedule_mode == "fixed":
            fixed_duration_s = float(self.config.box_fixed_duration_s)
            if fixed_duration_s > 0.0:
                return fixed_duration_s
            omega = max(abs(float(self.config.circle_omega_rad_s)), 1.0e-6)
            return 2.0 * math.pi / omega
        omega = max(abs(float(self.config.circle_omega_rad_s)), 1.0e-6)
        circle_period_s = 2.0 * math.pi / omega
        initial_open = max(0.0, float(self.config.box_initial_open_periods))
        shrink = max(1.0e-6, float(self.config.box_axis_shrink_periods))
        hold = max(0.0, float(self.config.box_axis_hold_periods))
        reopen = max(1.0e-6, float(self.config.box_axis_open_periods))
        between_axes = max(0.0, float(self.config.box_between_axes_open_periods))
        final_open = max(0.0, float(self.config.box_final_open_periods))
        axis_stage = shrink + hold + reopen
        return (initial_open + axis_stage + between_axes + axis_stage + final_open) * circle_period_s

    def _staged_half_extents(self, t_s: float) -> Tuple[float, float]:
        open_extent = float(self.config.box_l0_m)
        narrow_extent = min(open_extent, float(self.config.box_min_half_extent_m))
        if self.config.box_schedule_mode == "fixed":
            return narrow_extent, narrow_extent
        omega = max(abs(float(self.config.circle_omega_rad_s)), 1.0e-6)
        circle_period_s = 2.0 * math.pi / omega
        initial_open = max(0.0, float(self.config.box_initial_open_periods))
        shrink = max(1.0e-6, float(self.config.box_axis_shrink_periods))
        hold = max(0.0, float(self.config.box_axis_hold_periods))
        reopen = max(1.0e-6, float(self.config.box_axis_open_periods))
        between_axes = max(0.0, float(self.config.box_between_axes_open_periods))
        final_open = max(0.0, float(self.config.box_final_open_periods))
        axis_stage = shrink + hold + reopen
        cycle_periods = initial_open + axis_stage + between_axes + axis_stage + final_open
        phase = (float(t_s) % (cycle_periods * circle_period_s)) / circle_period_s

        lx = open_extent
        ly = open_extent
        x_start = initial_open
        y_start = initial_open + axis_stage + between_axes
        if x_start <= phase < x_start + axis_stage:
            lx = self._shrink_hold_open_extent(
                phase - x_start,
                open_extent,
                narrow_extent,
                shrink,
                hold,
                reopen,
            )
        elif y_start <= phase < y_start + axis_stage:
            ly = self._shrink_hold_open_extent(
                phase - y_start,
                open_extent,
                narrow_extent,
                shrink,
                hold,
                reopen,
            )
        return lx, ly

    @staticmethod
    def _shrink_hold_open_extent(
        stage_phase: float,
        open_extent: float,
        narrow_extent: float,
        shrink_periods: float,
        hold_periods: float,
        open_periods: float,
    ) -> float:
        if stage_phase < shrink_periods:
            progress = stage_phase / shrink_periods
            return open_extent + progress * (narrow_extent - open_extent)
        if stage_phase < shrink_periods + hold_periods:
            return narrow_extent
        progress = (stage_phase - shrink_periods - hold_periods) / open_periods
        return narrow_extent + progress * (open_extent - narrow_extent)

    def bounds_for(self, reference: ReferencePoint) -> np.ndarray:
        lx, ly = self._staged_half_extents(reference.t_s)
        x_ref = float(reference.position_m[0])
        y_ref = float(reference.position_m[1])
        box_center = self.config.circle_center_m
        world_x_min = float(box_center[0]) - lx
        world_x_max = float(box_center[0]) + lx
        world_y_min = float(box_center[1]) - ly
        world_y_max = float(box_center[1]) + ly
        return np.asarray(
            [
                x_ref - world_x_max,
                x_ref - world_x_min,
                y_ref - world_y_max,
                y_ref - world_y_min,
            ],
            dtype=np.float64,
        )


class BaseLoihiBackend:
    name = "base"

    def start(self, initial_state: np.ndarray) -> None:
        return None

    def solve(self, request: LoihiRequest) -> Optional[LoihiResult]:
        raise NotImplementedError

    def close(self) -> None:
        return None


class DisabledLoihiBackend(BaseLoihiBackend):
    name = "disabled"

    def solve(self, request: LoihiRequest) -> Optional[LoihiResult]:
        return None


class MockLoihiBackend(BaseLoihiBackend):
    name = "mock"

    def __init__(self, error_gain: float):
        self.error_gain = float(error_gain)

    def solve(self, request: LoihiRequest) -> Optional[LoihiResult]:
        start_s = time.monotonic()
        e_p1 = np.asarray(request.state_error[:3], dtype=np.float64) * self.error_gain
        e_p1[0] = float(np.clip(e_p1[0], request.bounds_error_xy[0], request.bounds_error_xy[1]))
        e_p1[1] = float(np.clip(e_p1[1], request.bounds_error_xy[2], request.bounds_error_xy[3]))
        return LoihiResult(
            output_epoch=int(request.epoch),
            u0=np.zeros(4, dtype=np.float64),
            selected_error_position_m=e_p1,
            latency_s=time.monotonic() - start_s,
            raw_output=None,
        )


class HostFloatLoihiBackend(BaseLoihiBackend):
    name = "host_float"

    def __init__(self, config: BridgeConfig):
        self.config = config
        self.problem_data = None
        self.A = None
        self.ATrho = None
        self.L = None
        self.z_state = None
        self.y_state = None
        self.lower = None
        self.upper = None
        self.state_dim = 0
        self.control_dim = 4
        self.equality_rows = 0
        self.input_inequality_rows = 0
        self.xy_inequality_rows = 0
        self.horizon = 0
        self.selected_position_index = 1
        self._prepare_import_path()

    def _prepare_import_path(self) -> Path:
        admm_path = Path(self.config.admm_nxcore_path).expanduser().resolve()
        if not admm_path.is_dir():
            raise RuntimeError(f"admm_nxcore path does not exist: {admm_path}")
        if str(admm_path) not in sys.path:
            sys.path.insert(0, str(admm_path))
        return admm_path

    def start(self, initial_state: np.ndarray) -> None:
        self._prepare_import_path()
        from admm_mpc.direct import infer_inequality_split
        from script.header_generator import build_default_bounds, get_solver_problem_data

        problem_data = get_solver_problem_data()
        self.problem_data = problem_data
        self.A = np.asarray(problem_data["A"], dtype=np.float64)
        self.ATrho = np.asarray(problem_data["ATrho"], dtype=np.float64)
        self.L = np.asarray(problem_data["L"], dtype=np.float64)
        self.state_dim = int(problem_data["n"])
        self.control_dim = int(problem_data["m"])
        self.equality_rows = int(problem_data["A_eq"].shape[0])
        (
            self.input_inequality_rows,
            self.xy_inequality_rows,
            self.horizon,
        ) = infer_inequality_split(
            self.A.shape[0],
            self.equality_rows,
            self.control_dim,
        )
        selected_index = int(self.config.loihi_selected_position_index)
        if selected_index < 0:
            selected_index = self.horizon
        if selected_index < 1 or selected_index > self.horizon:
            raise ValueError(
                "loihi_selected_position_index must be in 1..horizon or negative for terminal; "
                f"got {self.config.loihi_selected_position_index} with horizon={self.horizon}."
            )
        self.selected_position_index = selected_index

        initial_state = np.asarray(initial_state, dtype=np.float64)
        if initial_state.shape[0] != self.state_dim:
            raise ValueError(
                f"initial_state must have length {self.state_dim}, got {initial_state.shape[0]}"
            )
        self.lower, self.upper = build_default_bounds(np.zeros(self.state_dim, dtype=np.float64))
        self.z_state = np.zeros(self.A.shape[0], dtype=np.float64)
        self.y_state = np.zeros(self.A.shape[0], dtype=np.float64)
        self.z_state[: self.state_dim] = initial_state
        LOGGER.info(
            "Starting host-float ADMM backend: iterations=%d selected_position_index=%d horizon=%d",
            int(self.config.loihi_admm_iterations),
            self.selected_position_index,
            self.horizon,
        )

    def solve(self, request: LoihiRequest) -> Optional[LoihiResult]:
        if self.z_state is None or self.y_state is None:
            raise RuntimeError("Host-float backend has not been started.")
        start_s = time.monotonic()
        current_state = np.asarray(request.state_error, dtype=np.float64)
        if current_state.shape[0] != self.state_dim:
            raise ValueError(
                f"state_error must have length {self.state_dim}, got {current_state.shape[0]}"
            )

        lower = np.asarray(self.lower, dtype=np.float64).copy()
        upper = np.asarray(self.upper, dtype=np.float64).copy()
        xy_lower, xy_upper = self._expand_xy_bounds_float(request.bounds_error_xy)
        xy_start = self.equality_rows + self.input_inequality_rows
        xy_end = xy_start + self.xy_inequality_rows
        lower[xy_start:xy_end] = xy_lower
        upper[xy_start:xy_end] = xy_upper

        z_state = self.z_state.copy()
        y_state = self.y_state.copy()
        z_state[: self.state_dim] = current_state
        x = np.zeros(self.A.shape[1], dtype=np.float64)
        for _ in range(int(self.config.loihi_admm_iterations)):
            rhs = self.ATrho @ (z_state - y_state)
            w = np.linalg.solve(self.L, rhs)
            x = np.linalg.solve(self.L.T, w)
            v = self.A @ x + y_state
            z_next = np.empty_like(v)
            z_next[: self.state_dim] = current_state
            z_next[self.state_dim : self.equality_rows] = 0.0
            z_next[self.equality_rows :] = np.minimum(
                np.maximum(v[self.equality_rows :], lower[self.equality_rows :]),
                upper[self.equality_rows :],
            )
            y_state = v - z_next
            z_state = z_next

        self.z_state = z_state
        self.y_state = y_state
        u0_start = self.state_dim
        selected_start = self.selected_position_index * (self.state_dim + self.control_dim)
        return LoihiResult(
            output_epoch=int(request.epoch),
            u0=np.asarray(x[u0_start : u0_start + self.control_dim], dtype=np.float64),
            selected_error_position_m=np.asarray(
                x[selected_start : selected_start + 3],
                dtype=np.float64,
            ),
            latency_s=time.monotonic() - start_s,
            raw_output=None,
        )

    def _expand_xy_bounds_float(self, bounds_error_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        bounds = np.asarray(bounds_error_xy, dtype=np.float64)
        if bounds.shape[0] != 4:
            raise ValueError("xy bounds must be [x_min, x_max, y_min, y_max].")
        lower = np.empty(2 * self.horizon, dtype=np.float64)
        upper = np.empty(2 * self.horizon, dtype=np.float64)
        lower[0::2] = bounds[0]
        lower[1::2] = bounds[2]
        upper[0::2] = bounds[1]
        upper[1::2] = bounds[3]
        return lower, upper


class RealLoihiBackend(BaseLoihiBackend):
    name = "real"

    def __init__(self, config: BridgeConfig):
        self.config = config
        self.session = None
        self.vector_exp: Optional[int] = None
        self.problem_data = None
        self.control_dim = 4
        self._prepare_import_path()
        self._preload_runtime_modules()

    def _prepare_import_path(self) -> Path:
        admm_path = Path(self.config.admm_nxcore_path).expanduser().resolve()
        if not admm_path.is_dir():
            raise RuntimeError(f"admm_nxcore path does not exist: {admm_path}")
        if str(admm_path) not in sys.path:
            sys.path.insert(0, str(admm_path))
        return admm_path

    def _preload_runtime_modules(self) -> None:
        try:
            from nxcore.arch.n3b.n3board import N3Board  # noqa: F401
            from nxkernel.groups.eth_group import EthernetOutputServer  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "NxCore/NxKernel deep runtime imports failed before starting "
                "the Loihi backend. Run `make smoke` from the Docker host."
            ) from exc

    def start(self, initial_state: np.ndarray) -> None:
        admm_path = self._prepare_import_path()
        self._preload_runtime_modules()

        try:
            from admm_mpc import nx as admm_nx
        except Exception as exc:
            raise RuntimeError(
                "Could not import admm_mpc.nx from "
                f"{admm_path}; check that /intel/variables.sh eth and "
                "/intel/venv are active before launching the bridge."
            ) from exc
        if not admm_nx.has_loihi_runtime():
            raise RuntimeError(
                "admm_mpc.nx imported but HAS_LOIHI_RUNTIME is false. "
                "This usually means a nested NxCore/NxKernel import failed "
                f"inside {getattr(admm_nx, '__file__', '<unknown>')}. "
                "Run `make smoke` to expose the missing import directly."
            )

        from admm_mpc.core import choose_vector_exp, dequantize_vector, quantize_vector
        from admm_mpc.direct import build_admm_mpc_pipeline
        from admm_mpc.solver import StreamingMpcSession
        from script.header_generator import build_default_bounds, get_solver_problem_data

        self._dequantize_vector = dequantize_vector
        self._quantize_vector = quantize_vector

        problem_data = get_solver_problem_data()
        self.problem_data = problem_data
        self.control_dim = int(problem_data["m"])
        values = np.concatenate(
            [
                problem_data["u_min"],
                problem_data["u_max"],
                np.asarray([-4.4, 1.0, -1.0], dtype=np.float64),
            ]
        )
        max_abs_int = int(self.config.loihi_vector_max_abs_int)
        if max_abs_int > 0:
            vector_exp = choose_vector_exp(values, max_abs_int=max_abs_int)
        else:
            vector_exp = choose_vector_exp(values)
        self.vector_exp = int(vector_exp)

        n_state = int(problem_data["n"])
        initial_state = np.asarray(initial_state, dtype=np.float64)
        if initial_state.shape[0] != n_state:
            raise ValueError(f"initial_state must have length {n_state}, got {initial_state.shape[0]}")
        l, u = build_default_bounds(np.zeros(n_state, dtype=np.float64))
        solver_pipeline = build_admm_mpc_pipeline(
            problem_data,
            l,
            u,
            vector_exp,
            parallel_components=int(self.config.loihi_parallel_components),
            parallel_quant_bins=int(self.config.loihi_parallel_quant_bins),
            selected_position_index=int(self.config.loihi_selected_position_index),
            xy_dynamic_bounds_start_index=int(self.config.loihi_xy_dynamic_bounds_start_index),
        )
        self.selected_position_index = int(solver_pipeline.selected_position_index)

        y0_int = np.zeros(problem_data["A"].shape[0], dtype=np.int64)
        z0_int = np.zeros(problem_data["A"].shape[0], dtype=np.int64)
        initial_state_int = quantize_vector(initial_state, vector_exp)
        z0_int[:n_state] = initial_state_int
        preload_periods = 1 if bool(self.config.loihi_no_warm_start) else 0
        input_periods = int(self.config.loihi_admm_iterations) + preload_periods
        run_periods = max(1, int(self.config.loihi_max_control_ticks) * input_periods)

        LOGGER.info(
            "Starting StreamingMpcSession: ticks=%d admm_iterations=%d input_periods=%d vector_exp=%d selected_position_index=%d xy_dynamic_bounds_start_index=%d no_warm_start=%s",
            int(self.config.loihi_max_control_ticks),
            int(self.config.loihi_admm_iterations),
            input_periods,
            int(vector_exp),
            self.selected_position_index,
            int(self.config.loihi_xy_dynamic_bounds_start_index),
            bool(self.config.loihi_no_warm_start),
        )
        self.session = StreamingMpcSession(
            solver_pipeline,
            run_periods=run_periods,
            input_periods=input_periods,
            z_initial_state=z0_int,
            y_initial_state=y0_int,
            current_state_initial_state=initial_state_int,
            epoch_tagging=True,
            fresh_epoch_gate=True,
            dynamic_xy_bounds=True,
            ethernet_output_buffer_steps=int(self.config.loihi_ethernet_output_buffer_steps),
            selected_output_mode="selected_xyz",
            resident_cold_start=bool(self.config.loihi_no_warm_start),
        )

    def solve(self, request: LoihiRequest) -> Optional[LoihiResult]:
        if self.session is None or self.vector_exp is None:
            raise RuntimeError("Real Loihi backend has not been started.")
        state_int = self._quantize_vector(request.state_error, self.vector_exp)
        bounds_int = self._quantize_vector(request.bounds_error_xy, self.vector_exp)
        self.session.send_current_state(
            state_int,
            epoch=int(request.epoch),
            xy_bounds_int=bounds_int,
        )
        latest = self.session.recv_matching_output(
            int(request.epoch),
            timeout_s=float(self.config.loihi_match_timeout_s),
        )
        raw = np.asarray(latest["x"], dtype=np.int64)
        output_epoch = int(latest["epoch"])
        offset = 1
        if raw.shape[0] == 1 + 3:
            u0 = np.zeros(self.control_dim, dtype=np.float64)
            selected_error_position = self._dequantize_vector(raw[offset : offset + 3], self.vector_exp)
        else:
            u0 = self._dequantize_vector(raw[offset : offset + self.control_dim], self.vector_exp)
            selected_error_position = self._dequantize_vector(
                raw[offset + self.control_dim : offset + self.control_dim + 3],
                self.vector_exp,
            )
        latency_s = latest.get("state_to_output_latency_s")
        if latency_s is None:
            latency_s = time.monotonic() - request.sent_s
        return LoihiResult(
            output_epoch=output_epoch,
            u0=np.asarray(u0, dtype=np.float64),
            selected_error_position_m=np.asarray(selected_error_position, dtype=np.float64),
            latency_s=float(latency_s),
            raw_output=raw.copy(),
        )

    def close(self) -> None:
        if self.session is not None:
            self.session.close()
            self.session = None


class PositionCommandAdapter:
    def __init__(self, enabled: bool):
        self.enabled = bool(enabled)
        self.last_command: Optional[Command] = None

    def send(self, cf, command: Command) -> bool:
        self.last_command = command
        if not self.enabled:
            return False
        x, y, z = command.position_m.tolist()
        cf.commander.send_position_setpoint(x, y, z, math.degrees(command.yaw_rad))
        return True


class SafetySupervisor:
    def __init__(self, config: BridgeConfig):
        self.config = config
        self.fault_count = 0
        self.last_fault_reason = ""

    def note_success(self) -> None:
        self.fault_count = 0
        self.last_fault_reason = ""

    def note_fault(self, reason: str) -> None:
        self.fault_count += 1
        self.last_fault_reason = reason

    def should_land(self) -> bool:
        return self.fault_count >= int(self.config.fault_land_count)

    def clamp_command(
        self,
        proposed: Command,
        anchor_position_m: Optional[Sequence[float]],
        last_command: Optional[Command],
        z_min_m: Optional[float] = None,
    ) -> Command:
        position = np.asarray(proposed.position_m, dtype=np.float64).copy()
        z_min = self.config.z_min_m if z_min_m is None else float(z_min_m)
        position[2] = float(np.clip(position[2], z_min, self.config.z_max_m))

        if last_command is not None:
            anchor = np.asarray(last_command.position_m, dtype=np.float64)
        else:
            anchor = finite_or_none(anchor_position_m)

        if anchor is not None:
            delta = position - anchor
            norm = float(np.linalg.norm(delta))
            max_step = float(self.config.max_command_step_m)
            if max_step > 0.0 and norm > max_step:
                position = anchor + delta * (max_step / norm)
                position[2] = float(np.clip(position[2], z_min, self.config.z_max_m))

        return Command(position_m=position, yaw_rad=float(proposed.yaw_rad))


class BridgeCsvLogger:
    FIELDNAMES = [
        "wall_time",
        "monotonic_s",
        "mode",
        "backend",
        "commands_enabled",
        "command_sent",
        "fault_count",
        "fault_reason",
        "state_age_s",
        "mocap_age_s",
        "state_x",
        "state_y",
        "state_z",
        "state_vx",
        "state_vy",
        "state_vz",
        "state_qx",
        "state_qy",
        "state_qz",
        "state_qw",
        "state_omega_x",
        "state_omega_y",
        "state_omega_z",
        "ref_t_s",
        "ref_x",
        "ref_y",
        "ref_z",
        "ref_vx",
        "ref_vy",
        "ref_vz",
        "ref_yaw_deg",
        "ref_next_x",
        "ref_next_y",
        "ref_next_z",
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
        "bound_ex_min",
        "bound_ex_max",
        "bound_ey_min",
        "bound_ey_max",
        "request_epoch",
        "output_epoch",
        "selected_position_index",
        "loihi_latency_s",
        "u0_0",
        "u0_1",
        "u0_2",
        "u0_3",
        "selected_ep_x",
        "selected_ep_y",
        "selected_ep_z",
        "cmd_x",
        "cmd_y",
        "cmd_z",
        "cmd_yaw_deg",
    ]

    def __init__(self, path: str):
        if path:
            resolved = Path(path).expanduser()
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            resolved = Path(f"loihi_bridge_log_{ts}.csv")
        self.path = resolved
        self.file = self.path.open("w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=self.FIELDNAMES)
        self.writer.writeheader()
        self.file.flush()
        LOGGER.info("Writing Loihi bridge CSV log to %s", self.path)

    def write(self, row: Dict[str, object]) -> None:
        out = {name: row.get(name, "") for name in self.FIELDNAMES}
        self.writer.writerow(out)
        self.file.flush()

    def close(self) -> None:
        self.file.close()


class TerminalInput:
    def __init__(self):
        self.fd = None
        self.settings = None
        atexit.register(self.restore)

    def enable(self) -> None:
        if not sys.stdin.isatty():
            LOGGER.info("stdin is not a TTY; keyboard control disabled")
            return
        self.fd = sys.stdin.fileno()
        self.settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)

    def read_key(self) -> Optional[str]:
        if self.fd is None:
            return None
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    def restore(self) -> None:
        if self.fd is None or self.settings is None:
            return
        try:
            termios.tcsetattr(self.fd, termios.TCSANOW, self.settings)
        except termios.error:
            pass
        finally:
            self.fd = None
            self.settings = None


class CrazyflieLoihiBridge:
    MODE_IDLE = "IDLE"
    MODE_TAKEOFF = "TAKEOFF"
    MODE_HOLD = "HOLD"
    MODE_STAGE_DEMO = "STAGE_DEMO"
    MODE_DEMO = "LOIHI_DEMO"
    MODE_MANUAL = "LOIHI_MANUAL"
    MODE_LANDING = "LANDING"

    def __init__(self, config: BridgeConfig, mocap: MocapAdapter):
        self.config = config
        self.mocap = mocap
        self.shutdown_requested = threading.Event()
        self.state_adapter = StateEstimateZLogAdapter(config.state_log_period_ms)
        self.reference_generator = CircleReferenceGenerator(config)
        self.bounds_shifter = DynamicBoundsShifter(config)
        self.backend = self._make_backend(config)
        self.command_adapter = PositionCommandAdapter(config.position_commands_enabled)
        self.safety = SafetySupervisor(config)
        self.terminal = TerminalInput()
        self.csv_logger: Optional[BridgeCsvLogger] = None

        self.mode = self.MODE_IDLE
        self.hold_command: Optional[Command] = None
        self.demo_start_s: Optional[float] = None
        self.demo_yaw_hold_rad = math.radians(config.fixed_yaw_deg)
        self.manual_reference_position: Optional[np.ndarray] = None
        self.epoch = 0
        self.last_extpos_s = 0.0
        self.last_control_s = 0.0
        self.tick_count = 0
        self.quit_after_landing = False

    def _make_backend(self, config: BridgeConfig) -> BaseLoihiBackend:
        backend = config.loihi_backend.lower()
        if backend == "disabled":
            return DisabledLoihiBackend()
        if backend == "mock":
            return MockLoihiBackend(config.mock_error_gain)
        if backend == "host_float":
            return HostFloatLoihiBackend(config)
        if backend == "real":
            return RealLoihiBackend(config)
        raise ValueError("loihi_backend must be one of: disabled, mock, host_float, real")

    def request_shutdown(self) -> None:
        self.shutdown_requested.set()
        self.terminal.restore()

    def configure_crazyflie(self, cf) -> None:
        self.state_adapter.start(cf)
        cf.console.receivedChar.add_callback(lambda text: print(f"[CF_CONSOLE] {text}"))

        params = {
            "stabilizer.controller": 1,
            "stabilizer.estimator": 2,
            "kalman.resetEstimation": 1,
        }
        for name, value in params.items():
            LOGGER.info("Setting %s = %s", name, value)
            cf.param.set_value(name, value)
            time.sleep(0.1)
        cf.param.set_value("kalman.resetEstimation", 0)
        time.sleep(0.5)

        self.backend.start(np.zeros(12, dtype=np.float64))

        if self.config.arm_on_connect:
            cf.platform.send_arming_request(True)
            LOGGER.info("Sent Crazyflie arming request")

    def run(self, scf: SyncCrazyflie) -> None:
        cf = scf.cf
        self.csv_logger = BridgeCsvLogger(self.config.log_file)
        self.configure_crazyflie(cf)
        self.terminal.enable()
        self._print_controls()

        if self.config.auto_takeoff:
            self._start_takeoff()

        try:
            while not self.shutdown_requested.is_set():
                now_s = time.monotonic()
                key = self.terminal.read_key()
                if key is not None:
                    self._handle_key(key)

                if self.config.use_mocap and now_s - self.last_extpos_s >= self.config.extpos_period_s:
                    self._send_extpos(cf)
                    self.last_extpos_s = now_s

                if now_s - self.last_control_s >= self.config.control_period_s:
                    dt_s = (
                        self.config.control_period_s
                        if self.last_control_s <= 0.0
                        else now_s - self.last_control_s
                    )
                    self.last_control_s = now_s
                    self._control_tick(cf, now_s, dt_s)

                time.sleep(0.001)
        except KeyboardInterrupt:
            LOGGER.info("Control loop interrupted")
        finally:
            self.shutdown_requested.set()
            self.terminal.restore()
            try:
                cf.commander.send_stop_setpoint()
            except Exception:
                pass
            try:
                self.backend.close()
            except Exception as exc:
                LOGGER.warning("Loihi backend close failed: %r", exc)
            if self.csv_logger is not None:
                self.csv_logger.close()
                self.csv_logger = None
            LOGGER.info("Crazyflie Loihi bridge stopped")

    def _print_controls(self) -> None:
        LOGGER.info(
            "Controls: t=takeoff, n=manual Loihi, wasd=manual x/y, +/-=manual z, "
            "g=go to trajectory start, m=start Loihi trajectory, h=hold, l=land, q=land and quit"
        )
        LOGGER.info("Loihi backend: %s; position commands enabled: %s", self.backend.name, self.command_adapter.enabled)

    def _handle_key(self, key: str) -> None:
        if key == "t":
            self._start_takeoff()
        elif key == "n":
            self._start_manual_loihi()
        elif key == "g":
            self._start_stage_demo()
        elif key == "m":
            self._start_demo()
        elif key == "h":
            self._start_hold()
        elif key == "l":
            self._start_landing()
        elif key == "q":
            self._start_landing()
            self.quit_after_landing = True
        elif key in ("w", "a", "s", "d", "+", "=", "-", "_"):
            self._handle_manual_reference_key(key)

    def _send_extpos(self, cf) -> None:
        try:
            sent = self.mocap.send_external_pose(cf, self.state_adapter.latest())
            if not sent:
                return
        except Exception as exc:
            LOGGER.warning("Error sending mocap external position: %r", exc)

    def _start_takeoff(self) -> None:
        state = self.state_adapter.latest()
        anchor = self._current_position_or_last_command(state)
        yaw_rad = state.yaw_rad if state is not None else math.radians(self.config.fixed_yaw_deg)
        self.hold_command = Command(
            position_m=np.asarray([anchor[0], anchor[1], max(0.0, anchor[2])], dtype=np.float64),
            yaw_rad=yaw_rad,
        )
        self.demo_yaw_hold_rad = yaw_rad
        self.mode = self.MODE_TAKEOFF
        LOGGER.info("Takeoff requested from %.3f %.3f %.3f", *self.hold_command.position_m)

    def _start_hold(self) -> None:
        state = self.state_adapter.latest()
        anchor = self._current_position_or_last_command(state)
        yaw_rad = state.yaw_rad if state is not None else self.demo_yaw_hold_rad
        self.hold_command = Command(position_m=np.asarray(anchor, dtype=np.float64), yaw_rad=yaw_rad)
        self.demo_start_s = None
        self.manual_reference_position = None
        self.mode = self.MODE_HOLD
        LOGGER.info("Holding %.3f %.3f %.3f", *self.hold_command.position_m)

    def _start_demo(self) -> None:
        if self.backend.name == "disabled":
            LOGGER.warning("Cannot start Loihi demo while loihi_backend is disabled")
            return
        state = self.state_adapter.latest()
        if state is not None:
            self.demo_yaw_hold_rad = state.yaw_rad
        self.demo_start_s = time.monotonic()
        self.mode = self.MODE_DEMO
        LOGGER.info("Started Loihi circular-reference box demo")

    def _start_stage_demo(self) -> None:
        state = self.state_adapter.latest()
        yaw_rad = state.yaw_rad if state is not None else self.demo_yaw_hold_rad
        self.demo_yaw_hold_rad = yaw_rad
        reference = self.reference_generator.reference_at(0.0, yaw_rad)
        target = np.asarray(reference.position_m, dtype=np.float64).copy()
        target[2] = float(np.clip(target[2], self.config.z_min_m, self.config.z_max_m))
        self.hold_command = Command(position_m=target, yaw_rad=reference.yaw_rad)
        self.demo_start_s = None
        self.manual_reference_position = None
        self.mode = self.MODE_STAGE_DEMO
        LOGGER.info("Moving to Loihi demo start %.3f %.3f %.3f", *target)

    def _start_manual_loihi(self) -> None:
        if self.backend.name == "disabled":
            LOGGER.warning("Cannot start manual Loihi mode while loihi_backend is disabled")
            return
        state = self.state_adapter.latest()
        anchor = self._current_position_or_last_command(state)
        anchor[2] = float(np.clip(anchor[2], self.config.z_min_m, self.config.z_max_m))
        yaw_rad = state.yaw_rad if state is not None else self.demo_yaw_hold_rad
        self.demo_yaw_hold_rad = yaw_rad
        self.manual_reference_position = np.asarray(anchor, dtype=np.float64)
        self.demo_start_s = time.monotonic()
        self.mode = self.MODE_MANUAL
        LOGGER.info("Started manual Loihi mode at %.3f %.3f %.3f", *self.manual_reference_position)

    def _handle_manual_reference_key(self, key: str) -> None:
        if self.mode != self.MODE_MANUAL or self.manual_reference_position is None:
            return
        step = float(self.config.manual_step_m)
        if key == "w":
            self.manual_reference_position[0] += step
        elif key == "s":
            self.manual_reference_position[0] -= step
        elif key == "a":
            self.manual_reference_position[1] += step
        elif key == "d":
            self.manual_reference_position[1] -= step
        elif key in ("+", "="):
            self.manual_reference_position[2] += step
        elif key in ("-", "_"):
            self.manual_reference_position[2] -= step
        self.manual_reference_position[2] = float(
            np.clip(self.manual_reference_position[2], self.config.z_min_m, self.config.z_max_m)
        )
        LOGGER.info("Manual Loihi reference %.3f %.3f %.3f", *self.manual_reference_position)

    def _start_landing(self) -> None:
        if self.hold_command is None:
            state = self.state_adapter.latest()
            anchor = self._current_position_or_last_command(state)
            yaw_rad = state.yaw_rad if state is not None else self.demo_yaw_hold_rad
            self.hold_command = Command(np.asarray(anchor, dtype=np.float64), yaw_rad)
        self.demo_start_s = None
        self.manual_reference_position = None
        self.mode = self.MODE_LANDING
        LOGGER.info("Landing requested")

    def _control_tick(self, cf, now_s: float, dt_s: float) -> None:
        self.tick_count += 1
        state = self.state_adapter.latest()
        mocap_age = self.mocap.latest_age_s(now_s) if self.config.use_mocap else None
        state_age = self.state_adapter.latest_age_s(now_s)

        if self.config.auto_start_demo and self.mode == self.MODE_HOLD and self.backend.name != "disabled":
            self._start_demo()

        command: Optional[Command] = None
        command_sent = False
        row: Dict[str, object] = {
            "wall_time": datetime.now().isoformat(timespec="milliseconds"),
            "monotonic_s": f"{now_s:.6f}",
            "mode": self.mode,
            "backend": self.backend.name,
            "commands_enabled": int(self.command_adapter.enabled),
            "fault_count": self.safety.fault_count,
            "fault_reason": self.safety.last_fault_reason,
            "state_age_s": "" if state_age is None else f"{state_age:.6f}",
            "mocap_age_s": "" if mocap_age is None else f"{mocap_age:.6f}",
        }
        self._fill_state_row(row, state)

        active_mode = self.mode != self.MODE_IDLE
        safety_block_reason = ""
        if (
            active_mode
            and self.config.use_mocap
            and (mocap_age is None or mocap_age > self.config.max_mocap_age_s)
        ):
            safety_block_reason = "mocap_stale"
            self.safety.note_fault("mocap_stale")
            if self.safety.should_land():
                self._start_landing()
        elif self.mode in (self.MODE_DEMO, self.MODE_MANUAL) and (
            state_age is None or state_age > self.config.max_state_age_s
        ):
            safety_block_reason = "stateEstimateZ_stale"
            self.safety.note_fault("stateEstimateZ_stale")
            if self.safety.should_land():
                self._start_landing()

        if self.mode == self.MODE_IDLE:
            try:
                cf.commander.send_stop_setpoint()
            except Exception:
                pass
        elif safety_block_reason and self.mode != self.MODE_LANDING:
            command = self._fault_hold_command(state)
        elif self.mode == self.MODE_TAKEOFF:
            command = self._takeoff_command(dt_s)
            if command.position_m[2] >= self.config.takeoff_height_m:
                self.mode = self.MODE_HOLD
                LOGGER.info("Takeoff complete")
        elif self.mode == self.MODE_HOLD:
            command = self.hold_command
        elif self.mode == self.MODE_STAGE_DEMO:
            command = self._stage_demo_command()
        elif self.mode == self.MODE_LANDING:
            command = self._landing_command(dt_s)
        elif self.mode == self.MODE_DEMO:
            command, demo_row = self._demo_command(now_s, state)
            row.update(demo_row)
        elif self.mode == self.MODE_MANUAL:
            command, manual_row = self._manual_loihi_command(now_s, state)
            row.update(manual_row)

        if command is not None:
            anchor = None if state is None else state.position_m
            command = self.safety.clamp_command(
                command,
                anchor_position_m=anchor,
                last_command=self.command_adapter.last_command,
                z_min_m=0.0 if self.mode == self.MODE_LANDING else None,
            )
            command_sent = self.command_adapter.send(cf, command)
            row["cmd_x"] = f"{command.position_m[0]:.6f}"
            row["cmd_y"] = f"{command.position_m[1]:.6f}"
            row["cmd_z"] = f"{command.position_m[2]:.6f}"
            row["cmd_yaw_deg"] = f"{math.degrees(command.yaw_rad):.6f}"

        row["command_sent"] = int(command_sent)
        row["fault_count"] = self.safety.fault_count
        row["fault_reason"] = self.safety.last_fault_reason
        if self.csv_logger is not None:
            self.csv_logger.write(row)

        if self.tick_count % max(1, int(self.config.print_every)) == 0:
            self._print_status(row)

    def _takeoff_command(self, dt_s: float) -> Command:
        if self.hold_command is None:
            self._start_takeoff()
        assert self.hold_command is not None
        next_pos = self.hold_command.position_m.copy()
        next_pos[2] = min(
            float(self.config.takeoff_height_m),
            next_pos[2] + max(0.0, float(self.config.takeoff_rate_mps)) * max(0.0, dt_s),
        )
        self.hold_command = Command(position_m=next_pos, yaw_rad=self.hold_command.yaw_rad)
        return self.hold_command

    def _stage_demo_command(self) -> Command:
        reference = self.reference_generator.reference_at(0.0, self.demo_yaw_hold_rad)
        target = np.asarray(reference.position_m, dtype=np.float64).copy()
        target[2] = float(np.clip(target[2], self.config.z_min_m, self.config.z_max_m))
        self.hold_command = Command(position_m=target, yaw_rad=reference.yaw_rad)
        return self.hold_command

    def _landing_command(self, dt_s: float) -> Command:
        if self.hold_command is None:
            state = self.state_adapter.latest()
            anchor = self._current_position_or_last_command(state)
            self.hold_command = Command(np.asarray(anchor, dtype=np.float64), self.demo_yaw_hold_rad)
        next_pos = self.hold_command.position_m.copy()
        next_pos[2] = max(0.0, next_pos[2] - max(0.0, self.config.land_rate_mps) * max(0.0, dt_s))
        self.hold_command = Command(position_m=next_pos, yaw_rad=self.hold_command.yaw_rad)
        if next_pos[2] <= 0.02:
            self.mode = self.MODE_IDLE
            if self.quit_after_landing:
                self.shutdown_requested.set()
        return self.hold_command

    def _demo_command(
        self,
        now_s: float,
        state: Optional[FirmwareState],
    ) -> Tuple[Optional[Command], Dict[str, object]]:
        if state is None:
            self.safety.note_fault("stateEstimateZ_missing")
            if self.safety.should_land():
                self._start_landing()
            return self.hold_command, {}

        if self.demo_start_s is None:
            self.demo_start_s = now_s
        elapsed_s = now_s - self.demo_start_s
        demo_duration_s = self.bounds_shifter.cycle_period_s()
        if elapsed_s >= demo_duration_s:
            LOGGER.info("Loihi demo complete after %.2f s; switching to hold", elapsed_s)
            self._start_hold()
            return self.hold_command, {}
        reference_now = self.reference_generator.reference_at(elapsed_s, self.demo_yaw_hold_rad)
        reference_next = self.reference_generator.reference_at(
            elapsed_s + self.config.control_period_s * self._selected_reference_steps(),
            self.demo_yaw_hold_rad,
        )
        return self._loihi_reference_command(state, reference_now, reference_next)

    def _manual_loihi_command(
        self,
        now_s: float,
        state: Optional[FirmwareState],
    ) -> Tuple[Optional[Command], Dict[str, object]]:
        if state is None:
            self.safety.note_fault("stateEstimateZ_missing")
            if self.safety.should_land():
                self._start_landing()
            return self.hold_command, {}
        if self.manual_reference_position is None:
            self._start_manual_loihi()
        if self.manual_reference_position is None:
            return self.hold_command, {}
        elapsed_s = 0.0 if self.demo_start_s is None else now_s - self.demo_start_s
        reference = ReferencePoint(
            position_m=self.manual_reference_position.copy(),
            velocity_mps=np.zeros(3, dtype=np.float64),
            yaw_rad=self.demo_yaw_hold_rad,
            t_s=elapsed_s,
        )
        return self._loihi_reference_command(state, reference, reference)

    def _loihi_reference_command(
        self,
        state: FirmwareState,
        reference_now: ReferencePoint,
        reference_next: ReferencePoint,
    ) -> Tuple[Optional[Command], Dict[str, object]]:
        row: Dict[str, object] = {}
        bounds_error_xy = self.bounds_shifter.bounds_for(reference_next)
        state_error = self._build_loihi_state_error(state, reference_now)

        self.epoch += 1
        request = LoihiRequest(
            epoch=self.epoch,
            state_error=state_error,
            bounds_error_xy=bounds_error_xy,
            reference_now=reference_now,
            reference_next=reference_next,
            sent_s=time.monotonic(),
        )
        row.update(self._request_row(request))

        try:
            result = self.backend.solve(request)
        except Exception as exc:
            self.safety.note_fault(f"loihi_error:{exc}")
            LOGGER.warning("Loihi solve failed: %r", exc)
            if self.safety.should_land():
                self._start_landing()
            return self._hold_current_position(state), row

        if result is None:
            self.safety.note_fault("loihi_missing")
            if self.safety.should_land():
                self._start_landing()
            return self._hold_current_position(state), row

        row.update(self._result_row(result))
        if self.config.max_loihi_latency_s > 0.0 and result.latency_s > self.config.max_loihi_latency_s:
            self.safety.note_fault("loihi_stale")
            if self.safety.should_land():
                self._start_landing()
            return self._hold_current_position(state), row

        if int(result.output_epoch) != int(request.epoch):
            self.safety.note_fault("loihi_epoch_mismatch")
            if self.safety.should_land():
                self._start_landing()
            return self._hold_current_position(state), row

        p_cmd_abs = reference_next.position_m - result.selected_error_position_m
        command = Command(position_m=p_cmd_abs, yaw_rad=reference_next.yaw_rad)
        self.hold_command = command
        self.safety.note_success()
        return command, row

    def _build_loihi_state_error(self, state: FirmwareState, reference: ReferencePoint) -> np.ndarray:
        position_error = state.position_m - reference.position_m
        phi = rodrigues_attitude_error(state.quaternion_xyzw, reference.yaw_rad)
        return np.concatenate(
            [
                position_error,
                phi,
                state.velocity_mps,
                state.gyro_rad_s,
            ]
        ).astype(np.float64)

    def _selected_reference_steps(self) -> int:
        backend_index = int(
            getattr(
                self.backend,
                "selected_position_index",
                int(self.config.loihi_selected_position_index),
            )
        )
        return max(1, backend_index)

    def _hold_current_position(self, state: FirmwareState) -> Command:
        command = Command(position_m=state.position_m.copy(), yaw_rad=state.yaw_rad)
        self.hold_command = command
        return command

    def _fault_hold_command(self, state: Optional[FirmwareState]) -> Optional[Command]:
        if state is not None:
            return self._hold_current_position(state)
        return self.hold_command

    def _current_position_or_last_command(self, state: Optional[FirmwareState]) -> np.ndarray:
        if state is not None:
            return state.position_m.copy()
        mocap = self.mocap.latest()
        if mocap is not None:
            return mocap.position_m.copy()
        if self.command_adapter.last_command is not None:
            return self.command_adapter.last_command.position_m.copy()
        return np.asarray([0.0, 0.0, 0.0], dtype=np.float64)

    def _fill_state_row(self, row: Dict[str, object], state: Optional[FirmwareState]) -> None:
        if state is None:
            return
        row.update(
            {
                "state_x": f"{state.position_m[0]:.6f}",
                "state_y": f"{state.position_m[1]:.6f}",
                "state_z": f"{state.position_m[2]:.6f}",
                "state_vx": f"{state.velocity_mps[0]:.6f}",
                "state_vy": f"{state.velocity_mps[1]:.6f}",
                "state_vz": f"{state.velocity_mps[2]:.6f}",
                "state_qx": f"{state.quaternion_xyzw[0]:.9f}",
                "state_qy": f"{state.quaternion_xyzw[1]:.9f}",
                "state_qz": f"{state.quaternion_xyzw[2]:.9f}",
                "state_qw": f"{state.quaternion_xyzw[3]:.9f}",
                "state_omega_x": f"{state.gyro_rad_s[0]:.6f}",
                "state_omega_y": f"{state.gyro_rad_s[1]:.6f}",
                "state_omega_z": f"{state.gyro_rad_s[2]:.6f}",
            }
        )

    def _request_row(self, request: LoihiRequest) -> Dict[str, object]:
        e0 = request.state_error
        bounds = request.bounds_error_xy
        ref = request.reference_now
        ref_next = request.reference_next
        return {
            "ref_t_s": f"{ref.t_s:.6f}",
            "ref_x": f"{ref.position_m[0]:.6f}",
            "ref_y": f"{ref.position_m[1]:.6f}",
            "ref_z": f"{ref.position_m[2]:.6f}",
            "ref_vx": f"{ref.velocity_mps[0]:.6f}",
            "ref_vy": f"{ref.velocity_mps[1]:.6f}",
            "ref_vz": f"{ref.velocity_mps[2]:.6f}",
            "ref_yaw_deg": f"{math.degrees(ref.yaw_rad):.6f}",
            "ref_next_x": f"{ref_next.position_m[0]:.6f}",
            "ref_next_y": f"{ref_next.position_m[1]:.6f}",
            "ref_next_z": f"{ref_next.position_m[2]:.6f}",
            "e0_x": f"{e0[0]:.6f}",
            "e0_y": f"{e0[1]:.6f}",
            "e0_z": f"{e0[2]:.6f}",
            "e0_phi_x": f"{e0[3]:.6f}",
            "e0_phi_y": f"{e0[4]:.6f}",
            "e0_phi_z": f"{e0[5]:.6f}",
            "e0_vx": f"{e0[6]:.6f}",
            "e0_vy": f"{e0[7]:.6f}",
            "e0_vz": f"{e0[8]:.6f}",
            "e0_omega_x": f"{e0[9]:.6f}",
            "e0_omega_y": f"{e0[10]:.6f}",
            "e0_omega_z": f"{e0[11]:.6f}",
            "bound_ex_min": f"{bounds[0]:.6f}",
            "bound_ex_max": f"{bounds[1]:.6f}",
            "bound_ey_min": f"{bounds[2]:.6f}",
            "bound_ey_max": f"{bounds[3]:.6f}",
            "request_epoch": int(request.epoch),
        }

    def _result_row(self, result: LoihiResult) -> Dict[str, object]:
        u0 = np.zeros(4, dtype=np.float64)
        u0[: min(4, result.u0.shape[0])] = result.u0[: min(4, result.u0.shape[0])]
        return {
            "output_epoch": int(result.output_epoch),
            "selected_position_index": self._selected_reference_steps(),
            "loihi_latency_s": f"{result.latency_s:.6f}",
            "u0_0": f"{u0[0]:.6f}",
            "u0_1": f"{u0[1]:.6f}",
            "u0_2": f"{u0[2]:.6f}",
            "u0_3": f"{u0[3]:.6f}",
            "selected_ep_x": f"{result.selected_error_position_m[0]:.6f}",
            "selected_ep_y": f"{result.selected_error_position_m[1]:.6f}",
            "selected_ep_z": f"{result.selected_error_position_m[2]:.6f}",
        }

    def _print_status(self, row: Dict[str, object]) -> None:
        LOGGER.info(
            "mode=%s backend=%s epoch=%s/%s cmd=(%s,%s,%s) faults=%s reason=%s",
            row.get("mode", ""),
            row.get("backend", ""),
            row.get("request_epoch", ""),
            row.get("output_epoch", ""),
            row.get("cmd_x", ""),
            row.get("cmd_y", ""),
            row.get("cmd_z", ""),
            row.get("fault_count", ""),
            row.get("fault_reason", ""),
        )


class CrazyflieROS2Node(Node):
    def __init__(self):
        super().__init__("crazyflie_loihi_bridge_node")
        self._declare_parameters()
        self.config = self._load_config()
        self.mocap_adapter = MocapAdapter(
            self.config.axis_mapping,
            self.config.axis_sign,
            self.config.yaw_offset_rad,
            self.config.mocap_mode,
        )
        self.bridge = CrazyflieLoihiBridge(self.config, self.mocap_adapter)

        if self.config.use_mocap:
            mocap_topic = self.get_parameter("mocap_topic").value
            self.mocap_sub = self.create_subscription(
                PoseStamped,
                mocap_topic,
                self.mocap_callback,
                10,
            )
            self.get_logger().info(f"Subscribed to mocap topic: {mocap_topic}")
        else:
            self.get_logger().warning("MoCap disabled; external-position updates will not be sent")

        self._log_config()

    def _declare_parameters(self) -> None:
        self.declare_parameter("uri", URI)
        self.declare_parameter("mocap_topic", "/optitrack/marker/pose")
        self.declare_parameter("use_mocap", True)
        self.declare_parameter("mocap_mode", "position_only")
        self.declare_parameter("axis_mapping", [0, 1, 2])
        self.declare_parameter("axis_sign", [1.0, 1.0, 1.0])
        self.declare_parameter("yaw_offset_deg", 0.0)

        self.declare_parameter("control_period_s", 0.01)
        self.declare_parameter("extpos_period_s", 0.01)
        self.declare_parameter("state_log_period_ms", 10)
        self.declare_parameter("max_state_age_s", 0.1)
        self.declare_parameter("max_mocap_age_s", 0.25)
        self.declare_parameter("position_commands_enabled", True)
        self.declare_parameter("auto_takeoff", False)
        self.declare_parameter("auto_start_demo", False)
        self.declare_parameter("arm_on_connect", True)
        self.declare_parameter("takeoff_height_m", 0.5)
        self.declare_parameter("takeoff_rate_mps", 0.25)
        self.declare_parameter("land_rate_mps", 0.25)
        self.declare_parameter("z_min_m", 0.15)
        self.declare_parameter("z_max_m", 0.8)
        self.declare_parameter("max_command_step_m", 0.015)
        self.declare_parameter("fault_land_count", 10)

        self.declare_parameter("fixed_yaw_deg", 0.0)
        self.declare_parameter("reference_yaw_mode", "fixed")
        self.declare_parameter("circle_radius_m", DEFAULT_CIRCLE_RADIUS_M)
        self.declare_parameter("circle_omega_rad_s", DEFAULT_CIRCLE_OMEGA_RAD_S)
        self.declare_parameter("circle_center", DEFAULT_CIRCLE_CENTER_M)
        self.declare_parameter("box_l0_m", DEFAULT_BOX_OPEN_HALF_EXTENT_M)
        self.declare_parameter("box_min_half_extent_m", DEFAULT_BOX_NARROW_HALF_EXTENT_M)
        self.declare_parameter("box_initial_open_periods", DEFAULT_BOX_INITIAL_OPEN_PERIODS)
        self.declare_parameter("box_axis_shrink_periods", DEFAULT_BOX_AXIS_SHRINK_PERIODS)
        self.declare_parameter("box_axis_hold_periods", DEFAULT_BOX_AXIS_HOLD_PERIODS)
        self.declare_parameter("box_axis_open_periods", DEFAULT_BOX_AXIS_OPEN_PERIODS)
        self.declare_parameter("box_between_axes_open_periods", DEFAULT_BOX_BETWEEN_AXES_OPEN_PERIODS)
        self.declare_parameter("box_final_open_periods", DEFAULT_BOX_FINAL_OPEN_PERIODS)
        self.declare_parameter("box_schedule_mode", "staged")
        self.declare_parameter("box_fixed_duration_s", 30.0)
        self.declare_parameter("manual_step_m", 0.03)

        self.declare_parameter("loihi_backend", "disabled")
        self.declare_parameter("mock_error_gain", 0.0)
        self.declare_parameter("loihi_admm_iterations", 45)
        self.declare_parameter("loihi_parallel_components", 3)
        self.declare_parameter("loihi_parallel_quant_bins", 750)
        self.declare_parameter("loihi_vector_max_abs_int", -1)
        self.declare_parameter("loihi_max_control_ticks", 60000)
        self.declare_parameter("loihi_selected_position_index", -1)
        self.declare_parameter("loihi_xy_dynamic_bounds_start_index", 1)
        self.declare_parameter("loihi_no_warm_start", False)
        self.declare_parameter("loihi_match_timeout_s", 0.1)
        self.declare_parameter("max_loihi_latency_s", 0.1)
        self.declare_parameter("loihi_ethernet_output_buffer_steps", 4096)
        self.declare_parameter("admm_nxcore_path", default_admm_nxcore_path())
        self.declare_parameter("log_file", "")
        self.declare_parameter("print_every", 20)

    def _load_config(self) -> BridgeConfig:
        circle_center = list(self.get_parameter("circle_center").value)
        if len(circle_center) != 3:
            raise ValueError("circle_center must be [x, y, z].")
        return BridgeConfig(
            uri=str(self.get_parameter("uri").value),
            use_mocap=bool(self.get_parameter("use_mocap").value),
            mocap_mode=str(self.get_parameter("mocap_mode").value),
            axis_mapping=[int(value) for value in list(self.get_parameter("axis_mapping").value)],
            axis_sign=[float(value) for value in list(self.get_parameter("axis_sign").value)],
            yaw_offset_rad=math.radians(float(self.get_parameter("yaw_offset_deg").value)),
            control_period_s=float(self.get_parameter("control_period_s").value),
            extpos_period_s=float(self.get_parameter("extpos_period_s").value),
            state_log_period_ms=int(self.get_parameter("state_log_period_ms").value),
            max_state_age_s=float(self.get_parameter("max_state_age_s").value),
            max_mocap_age_s=float(self.get_parameter("max_mocap_age_s").value),
            position_commands_enabled=bool(self.get_parameter("position_commands_enabled").value),
            auto_takeoff=bool(self.get_parameter("auto_takeoff").value),
            auto_start_demo=bool(self.get_parameter("auto_start_demo").value),
            arm_on_connect=bool(self.get_parameter("arm_on_connect").value),
            takeoff_height_m=float(self.get_parameter("takeoff_height_m").value),
            takeoff_rate_mps=float(self.get_parameter("takeoff_rate_mps").value),
            land_rate_mps=float(self.get_parameter("land_rate_mps").value),
            z_min_m=float(self.get_parameter("z_min_m").value),
            z_max_m=float(self.get_parameter("z_max_m").value),
            max_command_step_m=float(self.get_parameter("max_command_step_m").value),
            fault_land_count=int(self.get_parameter("fault_land_count").value),
            fixed_yaw_deg=float(self.get_parameter("fixed_yaw_deg").value),
            reference_yaw_mode=str(self.get_parameter("reference_yaw_mode").value),
            circle_radius_m=float(self.get_parameter("circle_radius_m").value),
            circle_omega_rad_s=float(self.get_parameter("circle_omega_rad_s").value),
            circle_center_m=np.asarray(circle_center, dtype=np.float64),
            box_l0_m=float(self.get_parameter("box_l0_m").value),
            box_min_half_extent_m=float(self.get_parameter("box_min_half_extent_m").value),
            box_initial_open_periods=float(self.get_parameter("box_initial_open_periods").value),
            box_axis_shrink_periods=float(self.get_parameter("box_axis_shrink_periods").value),
            box_axis_hold_periods=float(self.get_parameter("box_axis_hold_periods").value),
            box_axis_open_periods=float(self.get_parameter("box_axis_open_periods").value),
            box_between_axes_open_periods=float(
                self.get_parameter("box_between_axes_open_periods").value
            ),
            box_final_open_periods=float(self.get_parameter("box_final_open_periods").value),
            box_schedule_mode=str(self.get_parameter("box_schedule_mode").value),
            box_fixed_duration_s=float(self.get_parameter("box_fixed_duration_s").value),
            manual_step_m=float(self.get_parameter("manual_step_m").value),
            loihi_backend=str(self.get_parameter("loihi_backend").value),
            mock_error_gain=float(self.get_parameter("mock_error_gain").value),
            loihi_admm_iterations=int(self.get_parameter("loihi_admm_iterations").value),
            loihi_parallel_components=int(self.get_parameter("loihi_parallel_components").value),
            loihi_parallel_quant_bins=int(self.get_parameter("loihi_parallel_quant_bins").value),
            loihi_vector_max_abs_int=int(self.get_parameter("loihi_vector_max_abs_int").value),
            loihi_max_control_ticks=int(self.get_parameter("loihi_max_control_ticks").value),
            loihi_selected_position_index=int(
                self.get_parameter("loihi_selected_position_index").value
            ),
            loihi_xy_dynamic_bounds_start_index=int(
                self.get_parameter("loihi_xy_dynamic_bounds_start_index").value
            ),
            loihi_no_warm_start=bool(self.get_parameter("loihi_no_warm_start").value),
            loihi_match_timeout_s=float(self.get_parameter("loihi_match_timeout_s").value),
            max_loihi_latency_s=float(self.get_parameter("max_loihi_latency_s").value),
            loihi_ethernet_output_buffer_steps=int(
                self.get_parameter("loihi_ethernet_output_buffer_steps").value
            ),
            admm_nxcore_path=str(self.get_parameter("admm_nxcore_path").value),
            log_file=str(self.get_parameter("log_file").value),
            print_every=int(self.get_parameter("print_every").value),
        )

    def _log_config(self) -> None:
        self.get_logger().info(f"Crazyflie URI: {self.config.uri}")
        self.get_logger().info(f"MoCap mode: {self.config.mocap_mode}")
        self.get_logger().info(f"Axis mapping: {self.config.axis_mapping}")
        self.get_logger().info(f"Axis sign: {self.config.axis_sign}")
        self.get_logger().info(f"Loihi backend: {self.config.loihi_backend}")
        self.get_logger().info(
            f"Position commands enabled: {self.config.position_commands_enabled}"
        )
        self.get_logger().info(f"admm_nxcore path: {self.config.admm_nxcore_path}")

    def mocap_callback(self, msg: PoseStamped) -> None:
        self.mocap_adapter.update(msg)

    def run(self) -> None:
        cflib.crtp.init_drivers()
        try:
            with SyncCrazyflie(self.config.uri, cf=Crazyflie(rw_cache="./cache")) as scf:
                self.get_logger().info("Connected to Crazyflie")
                self.bridge.run(scf)
        except Exception as exc:
            self.get_logger().error(f"Crazyflie bridge error: {exc}")
            traceback.print_exc()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CrazyflieROS2Node()
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt, shutting down")
    finally:
        node.bridge.request_shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        ros_thread.join(timeout=5.0)


if __name__ == "__main__":
    main()
