#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
from typing import Optional
from collections import deque

import torch

import rclpy
from rclpy.time import Time
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped, Pose

from tf2_ros import (
    Buffer,
    TransformListener,
    LookupException,
    ConnectivityException,
    ExtrapolationException,
)
from tf2_geometry_msgs import do_transform_pose_stamped

from robomimic.utils import file_utils as FileUtils

from goc_demo import robotiq


WORLD_FRAME = "world"

# --------------------------------------------------------------------------- #
#  Rotation helpers (adapted from real_world_data.py)                         #
# --------------------------------------------------------------------------- #

def _normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    return q / max(n, 1e-8)


def _quat_wxyz_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions in wxyz order."""
    left = _normalize_quat_wxyz(left)
    right = _normalize_quat_wxyz(right)
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return _normalize_quat_wxyz(np.array([
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ]))


def _rotvec_to_quat_wxyz(rotvec: np.ndarray) -> np.ndarray:
    """Rotation vector (axis-angle) → quaternion in wxyz order."""
    rotvec = np.asarray(rotvec, dtype=np.float64)
    angle = np.linalg.norm(rotvec)
    if angle < 1e-8:
        return np.array([1.0, 0.5 * rotvec[0], 0.5 * rotvec[1], 0.5 * rotvec[2]])
    axis = rotvec / angle
    half = 0.5 * angle
    return np.array([np.cos(half), *(np.sin(half) * axis)])


def _quat_wxyz_to_rot6d(q_wxyz: np.ndarray) -> np.ndarray:
    """Quaternion (wxyz) → first two columns of rotation matrix (6D)."""
    q = _normalize_quat_wxyz(q_wxyz)
    w, x, y, z = q
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r10 = 2.0 * (x * y + z * w)
    r20 = 2.0 * (x * z - y * w)
    r01 = 2.0 * (x * y - z * w)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r21 = 2.0 * (y * z + x * w)
    return np.array([r00, r10, r20, r01, r11, r21], dtype=np.float32)


def _ros_quat_to_wxyz(orientation) -> np.ndarray:
    """ROS orientation (xyzw) → wxyz array."""
    return np.array([orientation.w, orientation.x, orientation.y, orientation.z], dtype=np.float64)


# --------------------------------------------------------------------------- #
#  Node                                                                        #
# --------------------------------------------------------------------------- #

class DPInferenceNode(Node):

    # observation_horizon from the training config
    OBS_HORIZON = 2

    def __init__(self):
        super().__init__("dp_inference_node")

        # --- Parameters ---
        self.declare_parameter("pose_topic", "/cartesian_motion_controller/current_pose")
        self.declare_parameter("twist_topic", "/cartesian_motion_controller/current_twist")
        self.declare_parameter("obj_pose_topic", "/cheezit/pose")
        self.declare_parameter("ckpt_path", "/home/moritz/src/guided-diffusion/outputs/real_world/base_policy/model_best.pth")
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("gripper_mid", 128.5)  # raw gripper threshold: below→open(-1), above→closed(+1)

        self._pose_topic: str = self.get_parameter("pose_topic").value
        self._twist_topic: str = self.get_parameter("twist_topic").value
        self._obj_pose_topic: str = self.get_parameter("obj_pose_topic").value
        self._ckpt_path: str = self.get_parameter("ckpt_path").value
        self._rate_hz: float = float(self.get_parameter("rate_hz").value)
        self._gripper_mid: float = float(self.get_parameter("gripper_mid").value)

        if self._rate_hz <= 0.0:
            self.get_logger().warn("rate_hz must be > 0; defaulting to 10.0")
            self._rate_hz = 10.0

        # --- TF ---
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        # --- QoS ---
        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # --- Subscriptions ---
        self._latest_pose: Optional[Pose] = None
        self.create_subscription(PoseStamped, self._pose_topic, self._on_pose, best_effort_qos)

        self._latest_obj_pose: Optional[Pose] = None
        self.create_subscription(PoseStamped, self._obj_pose_topic, self._on_obj_pose, best_effort_qos)

        # --- Target pose publisher ---
        self.target_pose_publisher = self.create_publisher(
            PoseStamped, "/cartesian_motion_controller/target_frame", 10
        )

        # --- Gripper ---
        ip_address = "10.168.4.249"
        self._real_gripper = robotiq.RobotiqGripper(disabled=False)
        self._real_gripper.connect(ip_address, 63352)
        self._real_gripper.activate(auto_calibrate=True)
        self._real_gripper.open(speed=2, force=2)

        self._robot_paused = False
        self._pre_grasp_timer = None
        self._resume_timer = None
        self._pending_gripper_cmd = None
        self._grasp_settle_sec = 1.0
        self._grasp_pause_after_cmd_sec = 1.0

        # --- Observation history (frame stack = OBS_HORIZON) ---
        self._obs_deque: deque = deque(maxlen=self.OBS_HORIZON)
        self._last_commanded_gripper: float = -1.0  # track last commanded state to detect transitions

        # --- Load diffusion policy ---
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Loading checkpoint: {self._ckpt_path} on {device}")
        self._policy, _ = FileUtils.policy_from_checkpoint(
            ckpt_path=self._ckpt_path,
            device=device,
            verbose=True,
        )
        self._policy.start_episode()
        self.get_logger().info("Diffusion policy loaded.")

        # --- Timer ---
        self._timer = self.create_timer(1.0 / self._rate_hz, self._on_timer)
        self.get_logger().info(f"DP inference node running at {self._rate_hz:.1f} Hz")

    # ---------------------------------------------------------------------- #
    #  Callbacks                                                               #
    # ---------------------------------------------------------------------- #

    def _on_pose(self, msg: PoseStamped):
        ps_w = self._to_world(msg)
        if ps_w is not None:
            self._latest_pose = ps_w.pose

    def _on_obj_pose(self, msg: PoseStamped):
        ps_w = self._to_world(msg)
        if ps_w is not None:
            self._latest_obj_pose = ps_w.pose

    # ---------------------------------------------------------------------- #
    #  Main control loop                                                       #
    # ---------------------------------------------------------------------- #

    def _on_timer(self):
        if self._latest_pose is None:
            self.get_logger().info("Waiting for EEF pose…")
            return
        if self._latest_obj_pose is None:
            self.get_logger().info("Waiting for object pose…")
            return

        # ------------------------------------------------------------------ #
        #  Build observation dict for this timestep                           #
        # ------------------------------------------------------------------ #

        pose = self._latest_pose
        obj_pose = self._latest_obj_pose

        gripper_raw = float(self._real_gripper.get_current_position())
        gripper_binary = -1.0 if gripper_raw < self._gripper_mid else 1.0

        eef_pos = np.array([pose.position.x,
                             pose.position.y,
                             pose.position.z], dtype=np.float32)
        eef_rot6d = _quat_wxyz_to_rot6d(_ros_quat_to_wxyz(pose.orientation)).astype(np.float32)

        cheezit_pos = np.array([obj_pose.position.x,
                                 obj_pose.position.y,
                                 obj_pose.position.z], dtype=np.float32)
        cheezit_rot6d = _quat_wxyz_to_rot6d(_ros_quat_to_wxyz(obj_pose.orientation)).astype(np.float32)

        step_obs = {
            "eef_pos":       eef_pos,
            "eef_rot6d":     eef_rot6d,
            "gripper_binary": np.array([gripper_binary], dtype=np.float32),
            "cheezit_pos":   cheezit_pos,
            "cheezit_rot6d": cheezit_rot6d,
        }
        self._obs_deque.append(step_obs)

        # Pad deque with repeated first observation until full
        obs_list = list(self._obs_deque)
        while len(obs_list) < self.OBS_HORIZON:
            obs_list = [obs_list[0]] + obs_list

        # Stack along time axis: each value → [OBS_HORIZON, D]
        stacked_obs = {
            k: np.stack([o[k] for o in obs_list], axis=0)
            for k in step_obs
        }

        # ------------------------------------------------------------------ #
        #  Run diffusion policy inference                                      #
        # ------------------------------------------------------------------ #

        try:
            action = self._policy(stacked_obs)  # numpy array, shape (7,)
        except Exception as e:
            self.get_logger().error(f"Policy inference failed: {e}")
            return

        # action: [dx, dy, dz, drotvec_x, drotvec_y, drotvec_z, next_gripper_binary]
        dpos    = action[:3].astype(np.float64)
        drotvec = action[3:6].astype(np.float64)
        next_gripper = float(action[6])

        self.get_logger().info(
            f"action  dpos={dpos}  drotvec={drotvec}  gripper={next_gripper:.2f}"
        )

        # ------------------------------------------------------------------ #
        #  Compute next EEF pose                                               #
        # ------------------------------------------------------------------ #

        new_pos = np.array([pose.position.x,
                             pose.position.y,
                             pose.position.z], dtype=np.float64) + dpos

        # q_new = q_delta * q_current  (convention from training)
        q_cur_wxyz = _ros_quat_to_wxyz(pose.orientation)
        q_delta_wxyz = _rotvec_to_quat_wxyz(drotvec)
        q_new_wxyz = _quat_wxyz_multiply(q_delta_wxyz, q_cur_wxyz)

        # ------------------------------------------------------------------ #
        #  Publish target pose                                                 #
        # ------------------------------------------------------------------ #

        target = PoseStamped()
        target.header.frame_id = WORLD_FRAME
        target.header.stamp = self.get_clock().now().to_msg()
        target.pose.position.x = float(new_pos[0])
        target.pose.position.y = float(new_pos[1])
        target.pose.position.z = float(new_pos[2])
        target.pose.orientation.w = float(q_new_wxyz[0])
        target.pose.orientation.x = float(q_new_wxyz[1])
        target.pose.orientation.y = float(q_new_wxyz[2])
        target.pose.orientation.z = float(q_new_wxyz[3])

        if not self._robot_paused:
            self.target_pose_publisher.publish(target)

        # ------------------------------------------------------------------ #
        #  Execute gripper command on state transition                         #
        # ------------------------------------------------------------------ #

        prev = self._last_commanded_gripper
        if next_gripper > 0.0 and prev <= 0.0:
            self._last_commanded_gripper = 1.0
            self._pause_robot_delayed(
                pre_delay=self._grasp_settle_sec,
                post_delay=self._grasp_pause_after_cmd_sec,
                gripper_cmd="close",
            )
        elif next_gripper <= 0.0 and prev > 0.0:
            self._last_commanded_gripper = -1.0
            self._pause_robot_delayed(
                pre_delay=0.0,
                post_delay=0.0,
                gripper_cmd="open",
            )

    # ---------------------------------------------------------------------- #
    #  Gripper helpers                                                         #
    # ---------------------------------------------------------------------- #

    def _do_gripper_cmd(self, cmd: str):
        try:
            if cmd == "close":
                self._real_gripper.close(speed=200, force=2)
            elif cmd == "open":
                self._real_gripper.open(speed=200, force=2)
            else:
                self.get_logger().warn(f"Unknown gripper cmd: {cmd}")
        except Exception as e:
            self.get_logger().error(f"Gripper command '{cmd}' failed: {e}")

    def _on_pre_grasp(self):
        if self._pre_grasp_timer is not None:
            self._pre_grasp_timer.cancel()
            self._pre_grasp_timer = None
        cmd = self._pending_gripper_cmd
        self._pending_gripper_cmd = None
        if cmd is not None:
            self._do_gripper_cmd(cmd)
        if self._resume_timer is not None:
            self._resume_timer.cancel()
            self._resume_timer = None
        self._resume_timer = self.create_timer(
            self._grasp_pause_after_cmd_sec, self._resume_robot
        )

    def _resume_robot(self):
        self._robot_paused = False
        if self._resume_timer is not None:
            self._resume_timer.cancel()
            self._resume_timer = None
        self.get_logger().info("Robot motion resumed.")

    def _pause_robot_delayed(self, pre_delay: float, post_delay: float, gripper_cmd: str):
        self._robot_paused = True
        self._pending_gripper_cmd = gripper_cmd
        if self._pre_grasp_timer is not None:
            self._pre_grasp_timer.cancel()
            self._pre_grasp_timer = None
        self._pre_grasp_timer = self.create_timer(pre_delay, self._on_pre_grasp)
        if self._resume_timer is not None:
            self._resume_timer.cancel()
            self._resume_timer = None

    # ---------------------------------------------------------------------- #
    #  TF helpers                                                              #
    # ---------------------------------------------------------------------- #

    def _to_world(
        self,
        pose_msg: PoseStamped,
        timeout_sec: float = 0.1,
        target_frame: str = WORLD_FRAME,
    ) -> Optional[PoseStamped]:
        if pose_msg is None:
            return None
        src_frame = pose_msg.header.frame_id
        if not src_frame:
            self.get_logger().warn("Incoming PoseStamped has empty frame_id")
            return None
        if src_frame == target_frame:
            return pose_msg
        try:
            tf = self.tf_buffer.lookup_transform(
                target_frame,
                src_frame,
                Time(),
                timeout=rclpy.duration.Duration(seconds=timeout_sec),
            )
            return do_transform_pose_stamped(pose_msg, tf)
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(
                f"TF transform failed ({target_frame} ← {src_frame}): {e}"
            )
            return None


# --------------------------------------------------------------------------- #

def main(args=None):
    rclpy.init(args=args)
    node = DPInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
