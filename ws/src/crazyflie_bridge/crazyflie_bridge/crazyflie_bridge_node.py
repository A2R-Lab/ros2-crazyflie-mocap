#!/usr/bin/env python3

import csv
import logging
import os
import atexit
import time
import sys
import select
import termios
import tty
import math
import struct
from datetime import datetime
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import threading

# Crazyflie URI
URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')
DELTA = 0.10
def quaternion_to_yaw(qx, qy, qz, qw):
    """Extract yaw (rotation around Z) from quaternion in radians."""
    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def world_to_body_velocity(vx_world, vy_world, yaw):
    """
    Transform velocity from world frame to body frame.
    yaw: current heading in radians
    """
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    vx_body = cos_yaw * vx_world + sin_yaw * vy_world
    vy_body = -sin_yaw * vx_world + cos_yaw * vy_world
    return vx_body, vy_body


def body_to_world_displacement(dx_body, dy_body, yaw):
    """
    Transform displacement from body frame to world frame.
    dx_body: forward displacement in body frame
    dy_body: left displacement in body frame  
    yaw: current heading in radians
    """
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    dx_world = cos_yaw * dx_body - sin_yaw * dy_body
    dy_world = sin_yaw * dx_body + cos_yaw * dy_body
    return dx_world, dy_world


def apply_yaw_offset_to_quaternion(qx, qy, qz, qw, yaw_offset):
    """
    Apply a yaw (Z-axis) rotation offset to a quaternion.
    This is used to correct for rigid body creation orientation in OptiTrack.
    yaw_offset: rotation offset in radians
    """
    # Create rotation quaternion for yaw offset (rotation around Z)
    half_angle = yaw_offset / 2.0
    offset_qw = math.cos(half_angle)
    offset_qz = math.sin(half_angle)
    
    # Quaternion multiplication: q_result = q_offset * q_original
    # This rotates the original orientation by the offset
    new_qw = offset_qw * qw - offset_qz * qz
    new_qx = offset_qw * qx + offset_qz * qy
    new_qy = offset_qw * qy - offset_qz * qx
    new_qz = offset_qw * qz + offset_qz * qw
    
    return new_qx, new_qy, new_qz, new_qw


def yaw_to_quaternion(yaw):
    """
    Create a quaternion representing only yaw rotation (around Z axis).
    Roll and pitch are zero.
    yaw: heading in radians
    Returns: (qx, qy, qz, qw)
    """
    half_yaw = yaw / 2.0
    qx = 0.0
    qy = 0.0
    qz = math.sin(half_yaw)
    qw = math.cos(half_yaw)
    return qx, qy, qz, qw

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CrazyflieController:
    def __init__(self, uri, use_mocap=True, trajectory_file='', trajectory_offset=None):
        self.uri = uri
        self.scf = None
        self.is_flying = False
        self.target_height = 0.0
        self.state = 'IDLE'  # IDLE, TAKING_OFF, FLYING, LANDING, PLAYING_TRAJECTORY
        self.use_mocap = use_mocap
        self.trajectory_file = trajectory_file or 'trajectory.csv'
        self.trajectory_offset = trajectory_offset or [0.0, 0.0, 0.0, 0.0]  # [x, y, z, yaw_deg]
        
        # Target position for mocap mode
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 0.0
        self.target_yaw = 0.0
        
        # Current mocap pose (updated from external source)
        self.current_yaw = 0.0  # Current yaw from mocap in radians

        # Dynamic constraints from ROS node
        self.dynamic_xy_min_mm = 0
        self.dynamic_xy_max_mm = 0
        self.sending_constraints = False
        self.last_constraint_send_time = 0
        self.constraint_rate = 0.10  # 10Hz
        self.manual_constraints = False
        self.manual_constraint_step_mm = 50

        # Command constants
        self.CMD_MAGIC = 0xC0000000
        self.CMD_START_TRAJ = 1
        self.CMD_RESET_TRAJ = 2
        self.CMD_SET_CONSTRAINTS = 3

        # Trajectory playback: waypoints (relative), origin, start time, next index, mocap log file
        self.trajectory_waypoints = []
        self.trajectory_origin = None  # (x, y, z, yaw_deg) in world frame when trajectory started
        self.trajectory_start_time = 0.0
        self.trajectory_next_index = 0  # next CSV line to apply when its time is reached
        self.mocap_log_file = None
        self.shutdown_requested = threading.Event()
        self._terminal_fd = None
        self._terminal_settings = None
        atexit.register(self.restore_terminal)
        
        # Timing control
        self.last_extpos_time = 0
        self.last_setpoint_time = 0
        self.extpos_rate = 0.01  # 100Hz
        self.setpoint_rate = 0.04  # 25Hz
        
        # Logging
        self.log_data = {
            'stateEstimate.x': 0,
            'stateEstimate.y': 0,
            'stateEstimate.z': 0,
            'stabilizer.roll': 0,
            'stabilizer.pitch': 0,
            'stabilizer.yaw': 0,
            'range.zrange': 0,
        }
        
    def _param_callback(self, name, value):
        logger.info(f'Parameter {name} set to {value}')
    
    def load_trajectory(self, filepath):
        """
        Load trajectory from CSV: time_ms, x, y, z, yaw (time in ms from 0).
        x, y, z are in meters and yaw in degrees; all are RELATIVE (offsets from start).
        Returns list of (time_ms, dx, dy, dz, dyaw_deg) or empty list on error.
        """
        waypoints = []
        path = os.path.expanduser(filepath)
        if not os.path.isfile(path):
            logger.error(f"Trajectory file not found: {path}")
            return waypoints
        try:
            with open(path, newline='') as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row or row[0].strip().startswith('#'):
                        continue
                    if len(row) < 5:
                        continue
                    try:
                        t_ms = float(row[0].strip())
                        x = float(row[1].strip())
                        y = float(row[2].strip())
                        z = float(row[3].strip())
                        yaw = float(row[4].strip())
                        waypoints.append((t_ms, x, y, z, yaw))
                    except (ValueError, IndexError):
                        continue
            waypoints.sort(key=lambda w: w[0])
            logger.info(f"Loaded {len(waypoints)} waypoints from {path}")
        except Exception as e:
            logger.error(f"Failed to load trajectory: {e}")
        return waypoints
    
    def setup_logging(self, cf):
        """Setup logging configuration for state estimation"""
        log_conf = LogConfig(name='StateEstimate', period_in_ms=200)
        
        # Add variables to log
        log_conf.add_variable('stateEstimate.x', 'float')
        log_conf.add_variable('stateEstimate.y', 'float')
        log_conf.add_variable('stateEstimate.z', 'float')
        log_conf.add_variable('stabilizer.roll', 'float')
        log_conf.add_variable('stabilizer.pitch', 'float')
        log_conf.add_variable('stabilizer.yaw', 'float')
        log_conf.add_variable('range.zrange', 'uint16_t')  # Flow deck height
        
        try:
            cf.log.add_config(log_conf)
            log_conf.data_received_cb.add_callback(self._log_callback)
            log_conf.start()
            logger.info("Logging started")
        except KeyError as e:
            logger.error(f'Could not start log configuration: {e}')
        except AttributeError as e:
            logger.error(f'Could not add log config: {e}')
    
        
    def _console_callback(self, text):
        """Callback for console output from Crazyflie"""
        msg = f"[CF_CONSOLE] {text}"
        print(msg)

    def _enable_keyboard_input(self):
        """Put stdin in cbreak mode and save enough state to restore it later."""
        if not sys.stdin.isatty():
            logger.warning("stdin is not a TTY; keyboard control disabled")
            return

        self._terminal_fd = sys.stdin.fileno()
        self._terminal_settings = termios.tcgetattr(self._terminal_fd)
        tty.setcbreak(self._terminal_fd)

    def restore_terminal(self):
        """Restore stdin terminal settings once, even if shutdown comes from another thread."""
        if self._terminal_fd is None or self._terminal_settings is None:
            return

        try:
            termios.tcsetattr(self._terminal_fd, termios.TCSANOW, self._terminal_settings)
        except termios.error as e:
            logger.debug(f"Could not restore terminal settings: {e}")
        finally:
            self._terminal_fd = None
            self._terminal_settings = None

    def request_shutdown(self):
        """Ask the control loop to stop and immediately undo terminal input mode."""
        self.shutdown_requested.set()
        self.restore_terminal()
    
    def _send_multiplexed_command(self, cf, cmd_type, payload=0):
        """
        Encode a command into the position setpoint floats.
        x_bits: [Magic(2) | CmdType(2) | Reserved(12) | PayloadMin(16)]
        y_bits: [Magic(2) | Reserved(14) | PayloadMax(16)]
        """
        # Pack x_bits
        x_bits = self.CMD_MAGIC | (cmd_type << 28)
        if cmd_type == self.CMD_SET_CONSTRAINTS:
            x_bits |= (payload & 0xFFFF)

        # Pack y_bits
        y_bits = self.CMD_MAGIC
        if cmd_type == self.CMD_SET_CONSTRAINTS:
            y_bits |= ((payload >> 16) & 0xFFFF)

        # Convert bits to floats
        x_float = struct.unpack('f', struct.pack('I', x_bits))[0]
        y_float = struct.unpack('f', struct.pack('I', y_bits))[0]

        # Send magic setpoint
        cf.commander.send_position_setpoint(x_float, y_float, 0.0, 0.0)

    def _log_callback(self, timestamp, data, logconf):
        """Callback for logging data"""
        self.log_data.update(data)
        height_mm = data.get('range.zrange', 0)
        height_m = height_mm / 1000.0
        logger.info(f"[STATE] x={data['stateEstimate.x']:.3f} y={data['stateEstimate.y']:.3f} "
                   f"z={data['stateEstimate.z']:.3f} height={height_m:.3f}m | "
                   f"roll={data['stabilizer.roll']:.2f} pitch={data['stabilizer.pitch']:.2f} "
                   f"yaw={data['stabilizer.yaw']:.2f}")
        
    def set_parameters(self, cf, param_dict):
        """
        Set parameters on the Crazyflie
        param_dict: dictionary of parameter_name: value pairs
        """
        logger.info(f"Setting parameters: {param_dict}")
        
        for param_name, value in param_dict.items():
            try:
                cf.param.set_value(param_name, value)
                logger.info(f"Set {param_name} = {value}")
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Failed to set {param_name}: {e}")
                
    def run_control_loop(self, scf, mocap_func=None):
        """
        Main control loop
        Uses absolute position setpoints when mocap is available
        Uses velocity/hover commands for Flow Deck only
        """
        cf = scf.cf
        start_time = time.time()
        
        # Setup logging
        self.setup_logging(cf)
        cf.console.receivedChar.add_callback(self._console_callback)
        
        if self.use_mocap:
            logger.info("=" * 60)
            logger.info("MOCAP CONTROL MODE - BODY FRAME POSITIONING")
            logger.info("=" * 60)
            logger.info("IMPORTANT: Place drone facing world +X before starting!")
            logger.info("")
            logger.info("Commands:")
            logger.info("  t = takeoff (capture position and rise to 0.5m)")
            logger.info("  l = land")
            logger.info("  h = hover (stop at current position)")
            logger.info("  +/- = adjust target height")
            logger.info("  w/s = move forward/backward (body frame, 0.1m)")
            logger.info("  a/d = move left/right (body frame, 0.1m)")
            logger.info("  e/r = rotate yaw left/right (15 deg)")
            logger.info("  0 = reset yaw reference (drone must face +X)")
            logger.info("  g = go to trajectory zero point (trajectory_offset)")
            logger.info("  T = play trajectory from CSV (time_ms, dx, dy, dz, dyaw) using configured offset")
            logger.info("  O = start onboard FPGA trajectory")
            logger.info("  D = start onboard FPGA trajectory + enable dynamic constraints")
            logger.info("  B = enable dynamic constraints only (bounds relative to current setpoint)")
            logger.info("  8/2 = manual constraint test: move upper bound +/− 5cm")
            logger.info("  C = stop dynamic constraints (clear and revert to defaults)")
            logger.info("  R = reset onboard FPGA trajectory")
            logger.info("  1 = switch to PID controller")
            logger.info("  6 = switch to INDI controller")
            logger.info("  q = quit and land")
            logger.info("=" * 60)
        else:
            logger.info("=" * 60)
            logger.info("FLOW DECK CONTROL MODE")
            logger.info("=" * 60)
            logger.info("Commands:")
            logger.info("  t = takeoff")
            logger.info("  l = land")
            logger.info("  h = hover (stop moving)")
            logger.info("  +/- = adjust target height")
            logger.info("  w/s = forward/backward")
            logger.info("  a/d = left/right")
            logger.info("  1 = switch to PID controller")
            logger.info("  6 = switch to INDI controller")
            logger.info("  q = quit and land")
            logger.info("=" * 60)
        
        # Manual velocity control
        self.vx = 0.0
        self.vy = 0.0
        
        # Setup keyboard input (non-blocking)
        try:
            if self.shutdown_requested.is_set():
                return
            self._enable_keyboard_input()
            
            while not self.shutdown_requested.is_set():
                current_time = time.time()
                
                # Check for keyboard input (non-blocking)
                if self._terminal_fd is not None and select.select([sys.stdin], [], [], 0)[0]:
                    key = sys.stdin.read(1)
                    self._handle_keyboard(key, cf)
                    
                    if key == 'q':
                        logger.info("Quit command received - landing...")
                        self.state = 'LANDING'
                        self.target_height = 0.0
                        time.sleep(2)  # Give time to land
                        break
                
                # Send external pose/position if using mocap
                if self.use_mocap and mocap_func is not None:
                    if current_time - self.last_extpos_time >= self.extpos_rate:
                        try:
                            # Also get box constraints if available
                            x, y, z, qx, qy, qz, qw, mode, box_min, box_max = mocap_func()

                            # Update constraints in controller
                            if box_min is not None and box_max is not None and not self.manual_constraints:
                                self.dynamic_xy_min_mm = int(box_min * 1000.0)
                                self.dynamic_xy_max_mm = int(box_max * 1000.0)

                            if x > -25.0:  # Check for valid mocap data
                                # Log mocap to CSV while playing trajectory
                                if self.state == 'PLAYING_TRAJECTORY' and self.mocap_log_file is not None and self.trajectory_origin is not None:
                                    elapsed_ms = (current_time - self.trajectory_start_time) * 1000.0
                                    ox, oy, oz, oyaw = self.trajectory_origin
                                    yaw_rad = quaternion_to_yaw(qx, qy, qz, qw)
                                    yaw_deg = math.degrees(yaw_rad)
                                    rel_yaw = yaw_deg - oyaw
                                    if rel_yaw > 180:
                                        rel_yaw -= 360
                                    elif rel_yaw < -180:
                                        rel_yaw += 360
                                    sp_rel_yaw = self.target_yaw - oyaw
                                    if sp_rel_yaw > 180:
                                        sp_rel_yaw -= 360
                                    elif sp_rel_yaw < -180:
                                        sp_rel_yaw += 360
                                    self.mocap_log_file.write(
                                        f"{elapsed_ms:.2f},{x - ox:.6f},{y - oy:.6f},{z - oz:.6f},{rel_yaw:.4f},"
                                        f"{self.target_x - ox:.6f},{self.target_y - oy:.6f},{self.target_z - oz:.6f},"
                                        f"{sp_rel_yaw:.4f}\n"
                                    )
                                    self.mocap_log_file.flush()
                                if mode == 'position_only':
                                    # SINGLE MARKER MODE
                                    # Only send position - IMU handles all orientation
                                    # Get yaw from Crazyflie's IMU estimate, not mocap
                                    cf.extpos.send_extpos(x, y, z)
                                    self.current_yaw = math.radians(
                                        self.log_data.get('stabilizer.yaw', 0)
                                    )
                                elif mode == 'position_and_yaw':
                                    # Send position + yaw-only quaternion
                                    # Best compromise: precise position, yaw correction,
                                    # but IMU handles roll/pitch (smoother)
                                    self.current_yaw = quaternion_to_yaw(qx, qy, qz, qw)
                                    qx_yaw, qy_yaw, qz_yaw, qw_yaw = yaw_to_quaternion(self.current_yaw)
                                    cf.extpos.send_extpose(x, y, z, qx_yaw, qy_yaw, qz_yaw, qw_yaw)
                                else:  # 'full_pose'
                                    # Send full 6DoF pose from mocap
                                    self.current_yaw = quaternion_to_yaw(qx, qy, qz, qw)
                                    cf.extpos.send_extpose(x, y, z, qx, qy, qz, qw)
                                    
                            self.last_extpos_time = current_time
                        except Exception as e:
                            logger.error(f"Error sending extpose: {e}")
                
                # Handle periodic constraint sending
                if self.sending_constraints and (current_time - self.last_constraint_send_time >= self.constraint_rate):
                    payload = (self.dynamic_xy_min_mm & 0xFFFF) | ((self.dynamic_xy_max_mm & 0xFFFF) << 16)
                    self._send_multiplexed_command(cf, self.CMD_SET_CONSTRAINTS, payload)
                    self.last_constraint_send_time = current_time

                # Send setpoint at 50Hz based on state and mode
                if current_time - self.last_setpoint_time >= self.setpoint_rate:
                    try:
                        if self.use_mocap:
                            # MOCAP MODE - Use absolute position setpoints
                            if self.state == 'PLAYING_TRAJECTORY':
                                elapsed_ms = (current_time - self.trajectory_start_time) * 1000.0
                                ox, oy, oz, oyaw = self.trajectory_origin or (0, 0, 0, 0)
                                # Apply at most one waypoint per setpoint tick when its time is reached
                                # (so point 1 at 10s is applied after many 50Hz loops of mocap/setpoint, not in one batch)
                                if self.trajectory_next_index < len(self.trajectory_waypoints):
                                    t_ms, dx, dy, dz, dyaw_deg = self.trajectory_waypoints[self.trajectory_next_index]
                                    if elapsed_ms >= t_ms:
                                        self.target_x = ox + dx
                                        self.target_y = oy + dy
                                        self.target_z = oz + dz
                                        self.target_yaw = oyaw + dyaw_deg
                                        if self.target_yaw > 180:
                                            self.target_yaw -= 360
                                        elif self.target_yaw < -180:
                                            self.target_yaw += 360
                                        self.trajectory_next_index += 1
                                if self.trajectory_next_index >= len(self.trajectory_waypoints):
                                    self.state = 'FLYING'
                                    self.trajectory_origin = None
                                    self.trajectory_next_index = 0
                                    if self.mocap_log_file is not None:
                                        self.mocap_log_file.close()
                                        self.mocap_log_file = None
                                    logger.info("✓ Trajectory playback finished")
                            
                            if self.state == 'TAKING_OFF':
                                # Gradual takeoff to target height
                                if self.target_z < 0.5:
                                    self.target_z += 0.02
                                else:
                                    self.state = 'FLYING'
                                    logger.info(f"✓ Takeoff complete at ({self.target_x:.2f}, {self.target_y:.2f}, {self.target_z:.2f})")
                            
                            elif self.state == 'LANDING':
                                # Gradual landing
                                if self.target_z > 0.05:
                                    self.target_z -= 0.02
                                else:
                                    self.target_z = 0.0
                                    self.state = 'IDLE'
                                    logger.info("✓ Landing complete, now IDLE")
                            
                            # Constraint packets are multiplexed through the same
                            # commander setpoint queue. While they are streaming,
                            # do not overwrite them with normal position setpoints.
                            if self.state in ['TAKING_OFF', 'FLYING', 'LANDING', 'PLAYING_TRAJECTORY']:
                                if not self.sending_constraints:
                                    cf.commander.send_position_setpoint(
                                        self.target_x,
                                        self.target_y,
                                        self.target_z,
                                        self.target_yaw
                                    )
                            else:
                                cf.commander.send_stop_setpoint()
                        
                        else:
                            # FLOW DECK MODE - Use hover commands
                            if self.state == 'TAKING_OFF':
                                # Gradual takeoff - increase height
                                if self.target_height < 0.4:
                                    self.target_height += 0.02
                                else:
                                    self.state = 'FLYING'
                                    logger.info(f"✓ Takeoff complete, now FLYING at {self.target_height:.2f}m")
                            
                            elif self.state == 'LANDING':
                                # Gradual landing
                                if self.target_height > 0.05:
                                    self.target_height -= 0.02
                                else:
                                    self.target_height = 0.0
                                    self.state = 'IDLE'
                                    logger.info("✓ Landing complete, now IDLE")
                            
                            # Send appropriate command based on state
                            if self.state in ['TAKING_OFF', 'FLYING', 'LANDING']:
                                # Use hover mode for Flow Deck
                                cf.commander.send_hover_setpoint(
                                    self.vx,  # velocity x (m/s)
                                    self.vy,  # velocity y (m/s) 
                                    0.0,      # yaw rate (deg/s)
                                    self.target_height  # height above ground (m)
                                )
                            else:
                                # Send stop when idle
                                cf.commander.send_stop_setpoint()
                            
                        self.last_setpoint_time = current_time
                    except Exception as e:
                        logger.error(f"Error sending setpoint: {e}")
                
                # Small sleep to prevent busy waiting
                time.sleep(0.001)
                
        except KeyboardInterrupt:
            logger.info("Control loop interrupted")
        finally:
            self.shutdown_requested.set()
            if self.mocap_log_file is not None:
                try:
                    self.mocap_log_file.close()
                except OSError:
                    pass
                self.mocap_log_file = None
            # Restore terminal settings
            self.restore_terminal()
            # Send stop command
            try:
                cf.commander.send_stop_setpoint()
            except Exception as e:
                logger.debug(f"Could not send stop setpoint during shutdown: {e}")
            self.is_flying = False
            logger.info("Control loop stopped")

    def _enable_manual_constraints(self):
        """Use keyboard-controlled world-frame bounds instead of OptiTrack boxes."""
        if self.manual_constraints:
            return

        self.manual_constraints = True
        self.sending_constraints = True
        self.last_constraint_send_time = 0

        if self.dynamic_xy_min_mm == 0 and self.dynamic_xy_max_mm == 0:
            center_mm = int(self.target_x * 1000.0)
            self.dynamic_xy_min_mm = center_mm - 500
            self.dynamic_xy_max_mm = center_mm + 500

        logger.info(
            "Manual constraints enabled: "
            f"lower fixed={self.dynamic_xy_min_mm / 1000.0:.3f}m, "
            f"upper={self.dynamic_xy_max_mm / 1000.0:.3f}m"
        )
    
    def _handle_keyboard(self, key, cf):
        """Handle keyboard commands"""
        if key == 't':
            if self.state == 'IDLE':
                if self.use_mocap:
                    # Capture current position and yaw from state estimate
                    self.target_x = self.log_data.get('stateEstimate.x', 0)
                    self.target_y = self.log_data.get('stateEstimate.y', 0)
                    self.target_z = 0.0  # Start from ground
                    # Capture current yaw and use it as target (maintain heading)
                    self.target_yaw = math.degrees(self.current_yaw)
                    logger.info(f"⬆ TAKEOFF - position ({self.target_x:.2f}, {self.target_y:.2f}), yaw={self.target_yaw:.1f}°, rising to 0.5m")
                else:
                    logger.info("⬆ TAKEOFF command - starting takeoff")
                
                self.state = 'TAKING_OFF'
                self.target_height = 0.0
            else:
                logger.warning(f"⚠ Can only takeoff from IDLE state (current: {self.state})")
        
        elif key == 'l':
            if self.state in ['FLYING', 'TAKING_OFF']:
                logger.info("⬇ LAND command - starting landing")
                self.state = 'LANDING'
            else:
                logger.warning(f"⚠ Can only land from FLYING/TAKING_OFF state (current: {self.state})")
        
        elif key == 'h':
            if self.state == 'FLYING':
                if self.use_mocap:
                    # Capture current position and yaw, hold it
                    self.target_x = self.log_data.get('stateEstimate.x', self.target_x)
                    self.target_y = self.log_data.get('stateEstimate.y', self.target_y)
                    self.target_z = self.log_data.get('stateEstimate.z', self.target_z)
                    self.target_yaw = math.degrees(self.current_yaw)
                    logger.info(f"⏸ HOVER - locked at ({self.target_x:.2f}, {self.target_y:.2f}, {self.target_z:.2f}) yaw={self.target_yaw:.1f}°")
                else:
                    # Stop all velocity
                    self.vx = 0.0
                    self.vy = 0.0
                    logger.info(f"⏸ HOVER command - stopped at {self.target_height:.2f}m")
            else:
                logger.warning("⚠ Can only hover while FLYING")
        
        elif key == '+' or key == '=':
            if self.state == 'FLYING':
                if self.use_mocap:
                    self.target_z = min(self.target_z + DELTA, 2.0)
                    logger.info(f"⬆ Increasing height to {self.target_z:.2f}m")
                else:
                    self.target_height = min(self.target_height + DELTA, 1.5)
                    logger.info(f"⬆ Increasing height to {self.target_height:.2f}m")
        
        elif key == '-' or key == '_':
            if self.state == 'FLYING':
                if self.use_mocap:
                    self.target_z = max(self.target_z - DELTA, 0.2)
                    logger.info(f"⬇ Decreasing height to {self.target_z:.2f}m")
                else:
                    self.target_height = max(self.target_height - DELTA, 0.2)
                    logger.info(f"⬇ Decreasing height to {self.target_height:.2f}m")
        
        # Yaw control
        elif key == 'e':
            if self.state == 'FLYING' and self.use_mocap:
                self.target_yaw += 15.0  # Rotate left (counter-clockwise)
                # Normalize to -180 to 180
                if self.target_yaw > 180:
                    self.target_yaw -= 360
                logger.info(f"↺ Yaw left to {self.target_yaw:.1f}°")
        
        elif key == 'r':
            if self.state == 'FLYING' and self.use_mocap:
                self.target_yaw -= 15.0  # Rotate right (clockwise)
                # Normalize to -180 to 180
                if self.target_yaw < -180:
                    self.target_yaw += 360
                logger.info(f"↻ Yaw right to {self.target_yaw:.1f}°")
        
        # Controller switching
        elif key == '1':
            try:
                cf.param.set_value('stabilizer.controller', 1)
                logger.info("🎮 Switched to PID controller (1)")
            except Exception as e:
                logger.error(f"Failed to set controller: {e}")
        
        elif key == '6':
            try:
                cf.param.set_value('stabilizer.controller', 6)
                logger.info("🎮 Switched to FPGA controller (6)")
            except Exception as e:
                logger.error(f"Failed to set controller: {e}")
        
        # Go to trajectory zero point (g): the configured trajectory offset in world frame.
        elif key == 'g':
            if self.state == 'FLYING' and self.use_mocap:
                ox_off, oy_off, oz_off, oyaw_off = self.trajectory_offset
                self.target_x = ox_off
                self.target_y = oy_off
                self.target_z = oz_off
                self.target_yaw = oyaw_off
                while self.target_yaw > 180:
                    self.target_yaw -= 360
                while self.target_yaw < -180:
                    self.target_yaw += 360
                logger.info(
                    f"📍 Moved target to trajectory zero (offset): "
                    f"({self.target_x:.2f}, {self.target_y:.2f}, {self.target_z:.2f}), "
                    f"yaw={self.target_yaw:.1f}°"
                )
            else:
                logger.warning("⚠ Go to trajectory zero only when FLYING with mocap (press g)")

        # Trajectory playback from CSV (T = shift+t). Each point is offset by trajectory_offset.
        elif key == 'T':
            if self.state == 'FLYING' and self.use_mocap:
                self.trajectory_waypoints = self.load_trajectory(self.trajectory_file)
                if not self.trajectory_waypoints:
                    logger.warning("No trajectory loaded - check trajectory_file path")
                else:
                    self.trajectory_origin = tuple(self.trajectory_offset)
                    self.trajectory_next_index = 0
                    self.state = 'PLAYING_TRAJECTORY'
                    self.trajectory_start_time = time.time()
                    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                    mocap_log_path = f"mocap_log_{ts}.csv"
                    try:
                        self.mocap_log_file = open(mocap_log_path, 'w')
                        self.mocap_log_file.write(
                            "time_ms,x,y,z,yaw_deg,sp_x,sp_y,sp_z,sp_yaw_deg\n"
                        )
                        self.mocap_log_file.flush()
                        logger.info(
                            f"▶ Playing trajectory from {self.trajectory_file} "
                            f"(offset={self.trajectory_offset}), "
                            f"logging mocap+setpoint (relative) to {mocap_log_path}"
                        )
                    except OSError as e:
                        logger.error(f"Could not open mocap log file {mocap_log_path}: {e}")
                        self.mocap_log_file = None
            else:
                logger.warning("⚠ Start trajectory only when FLYING with mocap (press T)")

        # Onboard Trajectory Start (O)
        elif key == 'O':
            if self.state == 'FLYING' and self.use_mocap:
                logger.info("▶ Starting ONBOARD FPGA trajectory")
                self._send_multiplexed_command(cf, self.CMD_START_TRAJ)
            else:
                logger.warning("⚠ Start onboard trajectory only when FLYING with mocap (press O)")

        # Onboard Trajectory Start + Constraints (D)
        elif key == 'D':
            if self.state == 'FLYING' and self.use_mocap:
                logger.info("▶ Starting ONBOARD FPGA trajectory + DYNAMIC CONSTRAINTS")
                self._send_multiplexed_command(cf, self.CMD_START_TRAJ)
                self.sending_constraints = True
            else:
                logger.warning("⚠ Start onboard trajectory with constraints only when FLYING with mocap (press D)")

        # Dynamic Constraints Only (B)
        elif key == 'B':
            if self.state == 'FLYING' and self.use_mocap:
                self.manual_constraints = False
                logger.info("▣ Enabling DYNAMIC CONSTRAINTS without starting onboard trajectory")
                self.sending_constraints = True
                self.last_constraint_send_time = 0
            else:
                logger.warning("⚠ Enable dynamic constraints only when FLYING with mocap (press B)")

        # Manual dynamic constraint test: keep lower bound fixed, move upper bound.
        elif key == '8':
            if self.state == 'FLYING' and self.use_mocap:
                self._enable_manual_constraints()
                self.dynamic_xy_max_mm += self.manual_constraint_step_mm
                logger.info(
                    "Manual constraint upper increased: "
                    f"lower={self.dynamic_xy_min_mm / 1000.0:.3f}m, "
                    f"upper={self.dynamic_xy_max_mm / 1000.0:.3f}m"
                )
            else:
                logger.warning("⚠ Manual constraints only while FLYING with mocap")

        elif key == '2':
            if self.state == 'FLYING' and self.use_mocap:
                self._enable_manual_constraints()
                next_max = self.dynamic_xy_max_mm - self.manual_constraint_step_mm
                if next_max <= self.dynamic_xy_min_mm:
                    logger.warning("⚠ Manual constraint upper bound cannot cross lower bound")
                else:
                    self.dynamic_xy_max_mm = next_max
                    logger.info(
                        "Manual constraint upper decreased: "
                        f"lower={self.dynamic_xy_min_mm / 1000.0:.3f}m, "
                        f"upper={self.dynamic_xy_max_mm / 1000.0:.3f}m"
                    )
            else:
                logger.warning("⚠ Manual constraints only while FLYING with mocap")

        # Stop Constraints (C)
        elif key == 'C':
            logger.info("⏹ Stopping DYNAMIC CONSTRAINTS")
            self.manual_constraints = False
            self.sending_constraints = False
            self._send_multiplexed_command(cf, self.CMD_SET_CONSTRAINTS, 0) # Payload 0 clears

        # Reset Onboard Trajectory (R)
        elif key == 'R':
            logger.info("↺ Resetting ONBOARD FPGA trajectory")
            self._send_multiplexed_command(cf, self.CMD_RESET_TRAJ)
        
        # Yaw/Kalman reset - use when drone is facing +X to reset yaw reference
        elif key == '0':
            try:
                cf.param.set_value('kalman.resetEstimation', 1)
                time.sleep(0.1)
                cf.param.set_value('kalman.resetEstimation', 0)
                self.current_yaw = 0.0
                logger.info("🧭 Kalman filter reset - yaw reference set to 0 (drone should face +X)")
            except Exception as e:
                logger.error(f"Failed to reset Kalman filter: {e}")
        
        # Movement control - in BODY frame (relative to drone heading)
        elif key == 'w':
            if self.state == 'FLYING':
                if self.use_mocap:
                    # Move forward in body frame -> transform to world frame
                    dx_world, dy_world = body_to_world_displacement(DELTA, 0.0, self.current_yaw)
                    self.target_x += dx_world
                    self.target_y += dy_world
                    logger.info(f"↑ Forward (body) -> world ({self.target_x:.2f}, {self.target_y:.2f})")
                else:
                    self.vx = min(self.vx + DELTA, 0.5)
                    logger.info(f"→ Forward velocity: {self.vx:.2f} m/s")
        
        elif key == 's':
            if self.state == 'FLYING':
                if self.use_mocap:
                    # Move backward in body frame -> transform to world frame
                    dx_world, dy_world = body_to_world_displacement(-DELTA, 0.0, self.current_yaw)
                    self.target_x += dx_world
                    self.target_y += dy_world
                    logger.info(f"↓ Backward (body) -> world ({self.target_x:.2f}, {self.target_y:.2f})")
                else:
                    self.vx = max(self.vx - DELTA, -0.5)
                    logger.info(f"← Backward velocity: {self.vx:.2f} m/s")
        
        elif key == 'a':
            if self.state == 'FLYING':
                if self.use_mocap:
                    # Move left in body frame -> transform to world frame
                    dx_world, dy_world = body_to_world_displacement(0.0, DELTA, self.current_yaw)
                    self.target_x += dx_world
                    self.target_y += dy_world
                    logger.info(f"← Left (body) -> world ({self.target_x:.2f}, {self.target_y:.2f})")
                else:
                    self.vy = min(self.vy + DELTA, 0.5)
                    logger.info(f"← Left velocity: {self.vy:.2f} m/s")
        
        elif key == 'd':
            if self.state == 'FLYING':
                if self.use_mocap:
                    # Move right in body frame -> transform to world frame
                    dx_world, dy_world = body_to_world_displacement(0.0, -DELTA, self.current_yaw)
                    self.target_x += dx_world
                    self.target_y += dy_world
                    logger.info(f"→ Right (body) -> world ({self.target_x:.2f}, {self.target_y:.2f})")
                else:
                    self.vy = max(self.vy - DELTA, -0.5)
                    logger.info(f"→ Right velocity: {self.vy:.2f} m/s")
        
        elif key == 'q':
            logger.info("⏹ Quit requested")
        
        else:
            logger.debug(f"Unknown key: {key}")


class CrazyflieROS2Node(Node):
    def __init__(self):
        super().__init__('crazyflie_controller_node')
        
        # Declare parameters
        self.declare_parameter('uri', URI)
        self.declare_parameter('mocap_topic', '/optitrack/marker/pose')
        self.declare_parameter('box1_topic', '/optitrack/box1/pose')
        self.declare_parameter('box2_topic', '/optitrack/box2/pose')
        self.declare_parameter('use_mocap', True)
        
        # Coordinate frame transformation parameters
        # OptiTrack to Crazyflie frame mapping
        # Motive default is often: X-right, Y-up, Z-back (right-handed)
        # Crazyflie expects: X-forward, Y-left, Z-up (right-handed, ENU-like)
        # 
        # axis_mapping: which mocap axis maps to CF axis [cf_x, cf_y, cf_z]
        #   e.g., [0, 2, 1] means: CF_x = mocap_x, CF_y = mocap_z, CF_z = mocap_y
        # axis_sign: sign flip for each axis [sign_x, sign_y, sign_z]
        #   e.g., [1, -1, 1] means: flip Y axis
        self.declare_parameter('axis_mapping', [0, 1, 2])  # [x, y, z] -> [x, y, z] (no remapping)
        self.declare_parameter('axis_sign', [1.0, 1.0, 1.0])  # no sign flip
        
        # Yaw offset in degrees - to correct for rigid body creation orientation
        # If the drone's "front" in Motive doesn't match actual front, adjust here
        self.declare_parameter('yaw_offset_deg', 0.0)
        
        # What to send to Crazyflie from mocap:
        # 'position_only' - Only position, no orientation (IMU handles orientation, yaw may drift)
        # 'position_and_yaw' - Position + yaw only (best compromise: IMU for roll/pitch, mocap for yaw)
        # 'full_pose' - Full 6DoF pose (may be noisy for orientation)
        self.declare_parameter('mocap_mode', 'position_only')
        # Trajectory CSV path for playback (time_ms, x, y, z, yaw). Press T while flying to play.
        self.declare_parameter('trajectory_file', 'trajectory.csv')
        # Trajectory offset [x, y, z, yaw_deg] in world frame.
        # Each CSV point is applied as: world_target = trajectory_offset + trajectory_point.
        # Trajectory point (0, 0, 0, 0) is located at trajectory_offset.
        self.declare_parameter('trajectory_offset', [0.45, -0.55, 0.3, 0.0])
        # x=0.390 y=-0.582 z=0.517
        # Get parameters
        self.uri = self.get_parameter('uri').value
        self.use_mocap = self.get_parameter('use_mocap').value
        self.axis_mapping = list(self.get_parameter('axis_mapping').value)
        self.axis_sign = list(self.get_parameter('axis_sign').value)
        self.yaw_offset_rad = math.radians(self.get_parameter('yaw_offset_deg').value)
        self.mocap_mode = self.get_parameter('mocap_mode').value
        self.trajectory_file = self.get_parameter('trajectory_file').value
        self.trajectory_offset = list(self.get_parameter('trajectory_offset').value)
        if len(self.trajectory_offset) != 4:
            self.get_logger().warning(
                f"trajectory_offset must have 4 values [x, y, z, yaw_deg]; got {self.trajectory_offset}. "
                "Falling back to [0.0, 0.0, 0.0, 0.0]."
            )
            self.trajectory_offset = [0.0, 0.0, 0.0, 0.0]
        else:
            self.trajectory_offset = [float(v) for v in self.trajectory_offset]
        
        self.get_logger().info(f'URI: {self.uri}')
        self.get_logger().info(f'Use MoCap: {self.use_mocap}')
        self.get_logger().info(f'MoCap mode: {self.mocap_mode}')
        self.get_logger().info(f'Axis mapping: {self.axis_mapping}')
        self.get_logger().info(f'Axis sign: {self.axis_sign}')
        self.get_logger().info(f'Yaw offset: {math.degrees(self.yaw_offset_rad):.1f}°')
        self.get_logger().info(f'Trajectory offset [x, y, z, yaw_deg]: {self.trajectory_offset}')
        
        # Setup mocap if enabled
        self.mocap_pose = None
        self.mocap_lock = threading.Lock()
        self.last_box_log_time = 0.0
        
        if self.use_mocap:
            mocap_topic = self.get_parameter('mocap_topic').value
            self.mocap_sub = self.create_subscription(
                PoseStamped,
                mocap_topic,
                self.mocap_callback,
                10
            )

            self.box1_pose = None
            self.box2_pose = None
            self.box1_sub = self.create_subscription(
                PoseStamped,
                self.get_parameter('box1_topic').value,
                self.box1_callback,
                10
            )
            self.box2_sub = self.create_subscription(
                PoseStamped,
                self.get_parameter('box2_topic').value,
                self.box2_callback,
                10
            )
            self.get_logger().info(f'Subscribed to mocap topic: {mocap_topic}')
            self.get_logger().info(f"Subscribed to box topics")
        else:
            self.get_logger().info('MoCap disabled - using Flow Deck only')
        
        # Create controller
        self.controller = CrazyflieController(
            self.uri,
            use_mocap=self.use_mocap,
            trajectory_file=self.trajectory_file,
            trajectory_offset=self.trajectory_offset,
        )
        
    def mocap_callback(self, msg):
        """Callback for mocap pose data"""
        with self.mocap_lock:
            self.mocap_pose = msg
            # print("received pose from mocap", msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
    
    def box1_callback(self, msg):
        with self.mocap_lock:
            self.box1_pose = msg

    def box2_callback(self, msg):
        with self.mocap_lock:
            self.box2_pose = msg

    def get_mocap_data(self):
        """
        Get current mocap position data, transformed to Crazyflie frame.
        
        Applies:
        1. Axis remapping (axis_mapping parameter)
        2. Axis sign flipping (axis_sign parameter)
        3. Yaw offset correction (yaw_offset_deg parameter)
        
        Returns: (x, y, z, qx, qy, qz, qw, mocap_mode, box_min_x, box_max_x)
        """
        with self.mocap_lock:
            # Handle box constraints
            box_min = None
            box_max = None
            if self.box1_pose is not None and self.box2_pose is not None:
                box1_raw = [
                    self.box1_pose.pose.position.x,
                    self.box1_pose.pose.position.y,
                    self.box1_pose.pose.position.z
                ]
                box2_raw = [
                    self.box2_pose.pose.position.x,
                    self.box2_pose.pose.position.y,
                    self.box2_pose.pose.position.z
                ]
                # Transform box X coordinates using the same mapping as the drone
                b1_x = self.axis_sign[0] * box1_raw[self.axis_mapping[0]]
                b2_x = self.axis_sign[0] * box2_raw[self.axis_mapping[0]]
                box_min = min(b1_x, b2_x)
                box_max = max(b1_x, b2_x)
                now = time.time()
                if now - self.last_box_log_time >= 2.0:
                    self.get_logger().info(
                        "Boxes raw: "
                        f"box1=({box1_raw[0]:.3f}, {box1_raw[1]:.3f}, {box1_raw[2]:.3f}) m, "
                        f"box2=({box2_raw[0]:.3f}, {box2_raw[1]:.3f}, {box2_raw[2]:.3f}) m | "
                        f"constraint_x: box1={b1_x:.3f} m, box2={b2_x:.3f} m, "
                        f"min={box_min:.3f} m, max={box_max:.3f} m"
                    )
                    self.last_box_log_time = now

            if self.mocap_pose is None:
                return (-30.0, -30.0, 0.0, 0.0, 0.0, 0.0, 1.0, self.mocap_mode, box_min, box_max)  # Invalid data
            
            # Raw mocap position data
            pos_raw = [
                self.mocap_pose.pose.position.x,
                self.mocap_pose.pose.position.y,
                self.mocap_pose.pose.position.z
            ]
            
            # Apply axis mapping and sign to position
            # axis_mapping[i] tells which raw axis to use for CF axis i
            # axis_sign[i] tells the sign for CF axis i
            x = self.axis_sign[0] * pos_raw[self.axis_mapping[0]]
            y = self.axis_sign[1] * pos_raw[self.axis_mapping[1]]
            z = self.axis_sign[2] * pos_raw[self.axis_mapping[2]]
            
            # For position_only mode (single marker), skip quaternion processing
            # The quaternion from a single marker is meaningless
            # Yaw will come from Crazyflie's IMU instead
            if self.mocap_mode == 'position_only':
                return (x, y, z, 0.0, 0.0, 0.0, 1.0, self.mocap_mode, box_min, box_max)
            
            # Process quaternion for modes that use mocap orientation
            qx_raw = self.mocap_pose.pose.orientation.x
            qy_raw = self.mocap_pose.pose.orientation.y
            qz_raw = self.mocap_pose.pose.orientation.z
            qw_raw = self.mocap_pose.pose.orientation.w
            
            # Remap quaternion vector part (qx, qy, qz) same as position
            q_raw = [qx_raw, qy_raw, qz_raw]
            qx = self.axis_sign[0] * q_raw[self.axis_mapping[0]]
            qy = self.axis_sign[1] * q_raw[self.axis_mapping[1]]
            qz = self.axis_sign[2] * q_raw[self.axis_mapping[2]]
            qw = qw_raw
            
            # If we flipped an odd number of axes, we need to negate qw
            # to maintain a proper rotation (determinant = 1)
            sign_product = self.axis_sign[0] * self.axis_sign[1] * self.axis_sign[2]
            if sign_product < 0:
                qw = -qw
            
            # Apply yaw offset to correct for rigid body orientation in Motive
            if self.yaw_offset_rad != 0.0:
                qx, qy, qz, qw = apply_yaw_offset_to_quaternion(
                    qx, qy, qz, qw, self.yaw_offset_rad
                )
            
            return (x, y, z, qx, qy, qz, qw, self.mocap_mode, box_min, box_max)
    
    def run(self):
        """Main run function"""
        # Initialize cflib drivers
        cflib.crtp.init_drivers()
        
        try:
            with SyncCrazyflie(self.uri, cf=Crazyflie(rw_cache='./cache')) as scf:
                cf = scf.cf
                
                self.get_logger().info("✓ Connected to Crazyflie")
                time.sleep(1)
                
                # Set parameters for Flow Deck operation
                parameters = {
                    'stabilizer.controller': 1,  # 1 = PID controller
                    'kalman.resetEstimation': 1,
                }
                
                
                parameters['stabilizer.estimator'] = 2  # Kalman filter for mocap
                    
                
                self.controller.set_parameters(cf, parameters)
                time.sleep(1)
                
                cf.param.set_value('stabilizer.estimator', '2')
                

                # Reset the estimation
                cf.param.set_value('kalman.resetEstimation', '0')
                time.sleep(0.5)
                
                # Arm the motors
                cf.platform.send_arming_request(True)
                self.get_logger().info("✓ Armed - motors ready")
                time.sleep(1.0)
                
                # Run the control loop
                mocap_func = self.get_mocap_data if self.use_mocap else None
                self.controller.run_control_loop(scf, mocap_func=mocap_func)
                
                self.get_logger().info("✓ Flight completed")
                
        except Exception as e:
            self.get_logger().error(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()


def main(args=None):
    rclpy.init(args=args)
    
    node = CrazyflieROS2Node()
    
    # Run the crazyflie control in a separate thread
    # so ROS2 can still spin and receive mocap messages
    cf_thread = threading.Thread(target=node.run, daemon=True)
    cf_thread.start()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt, shutting down')
    finally:
        node.controller.request_shutdown()
        node.destroy_node()
        rclpy.shutdown()
        cf_thread.join(timeout=2)


if __name__ == '__main__':
    main()
