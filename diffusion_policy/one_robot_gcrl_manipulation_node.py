#!/usr/bin/env python3
from __future__ import annotations

import os
import argparse
import numpy as np
from typing import List, Optional, Tuple, Sequence, Union
from collections import namedtuple, defaultdict, deque

import pickle
from datetime import datetime

import rclpy
from rclpy.time import Time
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from rclpy.action import ActionClient
from sensor_msgs.msg import Image, JointState, PointCloud
from geometry_msgs.msg import PointStamped, PoseStamped, TwistStamped, Pose, Twist
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from nav_msgs.msg import Path
from control_msgs.action import FollowJointTrajectory
from builtin_interfaces.msg import Duration as RosDuration
from cv_bridge import CvBridge
import cv2

from tf2_ros import (
    Buffer,
    TransformListener,
    TransformException,
    LookupException,
    ConnectivityException,
    ExtrapolationException
)
from tf2_geometry_msgs import do_transform_pose_stamped, do_transform_point
from tf_transformations import quaternion_matrix, euler_from_quaternion, quaternion_from_euler

from pydrake.math import RollPitchYaw
from pydrake.common.eigen_geometry import Quaternion

# from goc_mpc.splines import Block
# from goc_mpc.goc_mpc import GraphOfConstraints, GraphOfConstraintsMPC
# from goc_mpc.simple_drake_env import SimpleDrakeGym

import importlib.util

import jax

from agents import agents as agent_registry
from utils.flax_utils import restore_agent
from utils.datasets import Dataset, GCDataset, HGCDataset, PixelHGCDataset, SequenceGCDataset

from goc_demo import robotiq

DATASET_CLASS_DICT = {
    'GCDataset':       GCDataset,
    'HGCDataset':      HGCDataset,
    'PixelHGCDataset': PixelHGCDataset,
    'SequenceGCDataset': SequenceGCDataset,
}


def _load_agent_config(config_path: str):
    """Import a GCRLManipulation agent config file and return its ConfigDict."""
    spec = importlib.util.spec_from_file_location("_agent_cfg", config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_config()


WORLD_FRAME = "world"


class OneRobotGCRLManipulationNode(Node):
    """
    """

    def __init__(self):
        super().__init__("one_robot_gcrl_manipulation_node")

        # --- Parameters (your snippet + a couple extra) ---
        self.declare_parameter("pose_topic", "/cartesian_motion_controller/current_pose")
        self.declare_parameter("twist_topic", "/cartesian_motion_controller/current_twist")
        self.declare_parameter("rate_hz", 30.0)
        self.declare_parameter("agent_config_path", "")
        self.declare_parameter("dataset_path", "")
        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("checkpoint_epoch", "")
        self.declare_parameter("obj_name", "")
        # Semicolon-separated key=value overrides, e.g. 'frame_stack=3;encoder=impala_small'.
        # Applied after loading the agent config file, mirroring --agent.key=value in training.
        # This format avoids YAML-special characters so it works unquoted on the ros2 CLI.
        self.declare_parameter("agent_config_overrides", "")

        self.bridge = CvBridge()
        self._target_img_dim = None  # set from dataset after loading

        # Read params
        self._pose_topic: str = self.get_parameter("pose_topic").value
        self._twist_topic: str = self.get_parameter("twist_topic").value
        self._rate_hz: float = float(self.get_parameter("rate_hz").value)

        if self._rate_hz <= 0.0:
            self.get_logger().warn("rate_hz must be > 0; defaulting to 30.0")
            self._rate_hz = 30.0

        self._period_sec = 1.0 / self._rate_hz

        # --- TF stuff ---
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        # --- Sub/Pub QoS ---
        pose_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        keypoints_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # --- Visualization Publications ---
        self._next_goal_publisher = self.create_publisher(Path, "/next_goal", 10)

        # --- Subscriptions ---
        self._latest_pose: Optional[PoseStamped] = None
        self.create_subscription(PoseStamped, self._pose_topic, self._on_pose, pose_qos)
        self._latest_twist: Optional[TwistStamped] = None
        self.create_subscription(TwistStamped, self._twist_topic, self._on_twist, pose_qos)

        self._latest_q: Optional[np.ndarray] = None
        self._latest_qd: Optional[np.ndarray] = None
        self._latest_eff: Optional[np.ndarray] = None
        self.create_subscription(JointState, "/joint_states", self._on_joints, pose_qos)

        self._latest_image = None
        self._obs_deque = None  # initialized after config load; subscription created conditionally

        self._latest_obj_pose: Optional[PoseStamped] = None
        self._obj_name: str = self.get_parameter("obj_name").value
        if self._obj_name:
            self.create_subscription(
                PoseStamped, f"{self._obj_name}/center_pose", self._on_obj_pose, pose_qos
            )

        # Publisher to send the target pose to the robot
        target_pose_topic_name = "/cartesian_motion_controller/target_frame"
        self.target_pose_publisher = self.create_publisher(
            PoseStamped, target_pose_topic_name, 10
        )

        # instatiate real grippers (not the cleanest, but has to be done)
        ip_address = "10.168.4.249"
        self._real_gripper = robotiq.RobotiqGripper(disabled=False)
        self._real_gripper.connect(ip_address, 63352)
        self._real_gripper.activate(auto_calibrate=True)
        self._real_gripper.open(speed=2, force=2)

        self._robot_paused = False
        self._pre_grasp_timer = None
        self._resume_timer = None

        # Pending gripper cmds (latched until pre-delay expires)
        self._pending_gripper_cmd = None
        self._last_commanded_gripper: float = -1.0  # -1 = open, +1 = closed

        # Tunables
        self._grasp_settle_sec = 1.00          # wait before actuating gripper
        self._grasp_pause_after_cmd_sec = 1.00 # time to remain paused after actuation

        # Initialize agent from ROS parameters.
        agent_config_path: str = self.get_parameter("agent_config_path").value
        dataset_path: str = self.get_parameter("dataset_path").value
        checkpoint_path: str = self.get_parameter("checkpoint_path").value
        checkpoint_epoch: str = self.get_parameter("checkpoint_epoch").value

        config = _load_agent_config(agent_config_path)

        # Apply overrides passed as a ROS parameter using semicolon-separated key=value pairs, e.g.:
        #   agent_config_overrides: 'frame_stack=3;encoder=impala_small;p_aug=0.5'
        # Values are parsed as Python literals (int, float, bool, str) via ast.literal_eval,
        # falling back to plain strings. This format avoids YAML-special characters so it
        # works unquoted on the ros2 CLI (-p agent_config_overrides:='frame_stack=3;...').
        overrides_str: str = self.get_parameter("agent_config_overrides").value
        if overrides_str:
            import ast
            overrides = {}
            for pair in overrides_str.split(";"):
                pair = pair.strip()
                if not pair:
                    continue
                key, _, val_str = pair.partition("=")
                key = key.strip()
                val_str = val_str.strip()
                try:
                    val = ast.literal_eval(val_str)
                except (ValueError, SyntaxError):
                    val = val_str  # treat as plain string
                overrides[key] = val
            config.update(overrides)
            self.get_logger().info(f"Applied agent_config_overrides: {overrides}")

        self._obs_mode = "image" if "Pixel" in config['dataset_class'] else "proprio"
        frame_stack = config.get('frame_stack') or 1  # config may store None; default to 1
        self.get_logger().info(f"obs_mode={self._obs_mode!r}, frame_stack={frame_stack}")

        if self._obs_mode == "image":
            self._obs_deque = deque(maxlen=frame_stack)
            self.create_subscription(Image, '/camera/camera/color/image_raw', self._on_image, 10)

        with open(dataset_path, "rb") as f:
            train_dataset = pickle.load(f)

        train_dataset.pop("ee_scale", None)
        self._action_min = train_dataset.pop("action_min", None)
        self._action_max = train_dataset.pop("action_max", None)
        self._action_norm = train_dataset.pop("actions_norm", None)

        if self._obs_mode == "image":
            self._target_img_dim = train_dataset["observations"].shape[1]
            self.get_logger().info(f"Image size from dataset: {self._target_img_dim}x{self._target_img_dim}")
        else:
            self._target_img_dim = None

        # action_min/max cover only the movement dims (gripper is not normalized).
        # shape (3,) : XYZ only
        # shape (4,) : XYZ + yaw.
        self._needs_yaw = (self._action_min is not None and self._action_min.shape[0] == 4)
        self.get_logger().info(
            f"Action mode: {'XYZ + yaw' if self._needs_yaw else 'XYZ only'} "
            f"(action_min dim = {self._action_min.shape[0] if self._action_min is not None else 'unknown'})"
        )

        train_dataset['terminals'][-1] = 1.0

        dataset_class = DATASET_CLASS_DICT[config['dataset_class']]
        train_dataset = dataset_class(Dataset.create(**train_dataset), config)

        goal_index = np.argwhere(train_dataset.dataset["terminals"])[0].item()
        goal_obs = train_dataset.dataset["observations"][goal_index]
        if self._obs_mode == "image" and frame_stack > 1:
            goal_obs = np.concatenate([goal_obs] * frame_stack, axis=-1)
        self._goal_obs = goal_obs # np.expand_dims(goal_obs, 0)

        example_batch = train_dataset.sample(1)

        agent_name = config['agent_name']
        agent_class = agent_registry[agent_name]

        if agent_name in ["gcfbc", "gcfql"]:
            agent = agent_class.create(
                0,
                example_batch,
                config,
            )
        else:
            agent = agent_class.create(
                0,
                example_batch,
                config,
                "real_world_experiment",
            )

        self.agent = restore_agent(agent, checkpoint_path, checkpoint_epoch)

        # --- Timing ---
        self._start_time = self.get_clock().now()
        self.end_elapsed_time = None
        self._timer = self.create_timer(self._period_sec, self._on_timer)

        self.get_logger().info(
            f"Streaming pose goals at {self._rate_hz:.1f} Hz"
        )

    def _denormalize_actions(self, actions_norm):
        denom = self._action_max - self._action_min
        denom = np.where(denom == 0, 1.0, denom)
        actions = self._action_min + 0.5 * (actions_norm + 1.0) * denom
        return actions

    # --- Callbacks ---
    def _on_joints(self, msg: JointState):
        self._latest_q = np.array(msg.position)
        self._latest_qd = np.array(msg.velocity)
        self._latest_eff = np.array(msg.effort)

    def _on_pose(self, msg: PoseStamped):
        ps_w = self._to_world(msg)
        if ps_w is not None:
            self._latest_pose = ps_w.pose

    def _on_twist(self, msg: TwistStamped):
        tw = self._twist_to_world(msg)
        if tw is not None:
            self._latest_twist = tw

    def _on_obj_pose(self, msg: PoseStamped):
        ps_w = self._to_world(msg)
        if ps_w is not None:
            self._latest_obj_pose = ps_w.pose

    def _on_image(self, msg):
        # 1. Convert ROS Image message to OpenCV format
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w = cv_img.shape[:2]

        # 2. Calculate cropping coordinates for a centered square
        # We find the shortest side to ensure the square fits
        size = min(h, w)
        start_x = (w - size) // 2
        start_y = (h - size) // 2

        # Crop using NumPy slicing: [y1:y2, x1:x2]
        square_crop = cv_img[start_y:start_y+size, start_x:start_x+size]

        # 3. Downscale to customizable resolution
        resized_img = cv2.resize(square_crop, (self._target_img_dim, self._target_img_dim), interpolation=cv2.INTER_AREA)

        self._latest_image = resized_img

    def _extract_state(self,
                       q: np.ndarray, # qd: np.ndarray, eff: np.ndarray, 
                       pose: Pose,
                       # twist: Twist,
                       gripper_pos: int) -> Tuple[np.ndarray, np.ndarray]:

        def pose_to_arr(pose: Pose):
            arr = np.array([pose.position.x,
                            pose.position.y,
                            pose.position.z])
            if self._needs_yaw:
                arr = np.append(arr, self._quat_to_yaw(pose.orientation))
            return arr

        # def twist_to_arr(twist: Twist):
        #     return np.array([twist.linear.x,
        #                      twist.linear.y,
        #                      twist.linear.z])

        parts = [# q, qd, eff,
                 pose_to_arr(pose),
                 # twist_to_arr(twist),
                 np.array([gripper_pos], dtype=float)]

        if self._latest_obj_pose is not None:
            parts.append(pose_to_arr(self._latest_obj_pose))

        return np.concatenate(parts)

    def _on_timer(self):
        if self._latest_pose is None:
            self.get_logger().info('_latest_pose is None')
            return
        if self._latest_twist is None:
            self.get_logger().info('_latest_twist is None')
            return

        if self._latest_q is None:
            self.get_logger().info('_latest_q is None')
            return
        if self._latest_qd is None:
            self.get_logger().info('_latest_qd is None')
            return
        if self._latest_eff is None:
            self.get_logger().info('_latest_eff is None')
            return

        if self._obs_mode == "image" and self._latest_image is None:
            self.get_logger().info('_latest_image is None')
            return

        if self._obj_name and self._latest_obj_pose is None:
            self.get_logger().info('_latest_obj_pose is None')
            return

        now = self.get_clock().now()
        t = (now - self._start_time).nanoseconds * 1e-9

        #######################################################################
        #                           GET OBSERVATION                           #
        #######################################################################

        try:
            proprioception = self._extract_state(self._latest_q,
                                                 # self._latest_qd,
                                                 # self._latest_eff,
                                                 self._latest_pose,
                                                 # self._latest_twist,
                                                 self._real_gripper.get_current_position())
            if self._obs_mode == "image":
                self._obs_deque.append(self._latest_image)
                fs = self._obs_deque.maxlen
                assert len(self._obs_deque) == fs, f"Need {fs} latest observations. Skipping."
                observation = np.concatenate(list(self._obs_deque), axis=-1)
            else:
                observation = proprioception
        except Exception as e:
            self.get_logger().warn(f"Bad State: {e}")
            return

        #######################################################################
        #                             AGENT STEP                              #
        #######################################################################

        if self._obs_mode == "image":
            action_norm = self.agent.sample_actions(observation, proprioception,
                                                    goals=self._goal_obs, seed=jax.random.PRNGKey(0))
        else:
            action_norm = self.agent.sample_actions(observation,
                                                    goals=self._goal_obs, seed=jax.random.PRNGKey(0))
        action_norm = np.asarray(action_norm)

        if self._needs_yaw:
            delta_norm    = action_norm[:4]   # [dx, dy, dz, dyaw]  — all normalized
            gripper_target = action_norm[4:]  # [gripper]
        else:
            delta_norm    = action_norm[:3]   # [dx, dy, dz]
            gripper_target = action_norm[3:]  # [gripper]

        delta = self._denormalize_actions(delta_norm)

        target_xyz = delta[:3] + np.array([self._latest_pose.position.x,
                                           self._latest_pose.position.y,
                                           self._latest_pose.position.z])

        if self._needs_yaw:
            current_yaw = self._quat_to_yaw(self._latest_pose.orientation)
            target_yaw  = delta[3] + current_yaw
            self.get_logger().info(f"delta_xyz: {delta_norm[:3]}, delta_yaw: {delta_norm[3]:.3f}, gripper: {gripper_target}")
            # self.get_logger().info(f"delta_xyz: {delta[:3]}, delta_yaw: {delta[3]:.3f}, gripper: {gripper_target}")
        else:
            self.get_logger().info(f"delta: {delta_norm}, gripper: {gripper_target}")

        #######################################################################
        #                            EXECUTE ACTION                           #
        #######################################################################

        target_pose_stamped = PoseStamped()
        target_pose_stamped.header.frame_id = WORLD_FRAME
        target_pose_stamped.header.stamp = self.get_clock().now().to_msg()
        target_pose_stamped.pose.position.x = float(target_xyz[0])
        target_pose_stamped.pose.position.y = float(target_xyz[1])
        target_pose_stamped.pose.position.z = float(target_xyz[2])

        if self._needs_yaw:
            qw, qx, qy, qz = self._yaw_to_quat(target_yaw)
            target_pose_stamped.pose.orientation.w = qw
            target_pose_stamped.pose.orientation.x = qx
            target_pose_stamped.pose.orientation.y = qy
            target_pose_stamped.pose.orientation.z = qz
        else:
            target_pose_stamped.pose.orientation.w = 0.0
            target_pose_stamped.pose.orientation.x = 0.0
            target_pose_stamped.pose.orientation.y = 1.0
            target_pose_stamped.pose.orientation.z = 0.0

        # qpos = np.concatenate((left_target_pose, right_target_pose))
        # self._obs, _, _, _, _ = self._env.step(qpos, grasp_cmds=self.goc_mpc.last_grasp_commands)

        # if len(self.goc_mpc.last_grasp_commands) > 0:
        #     self.get_logger().info(f"Grasp Commands! {self.goc_mpc.last_grasp_commands}")
        #     for cmd, robot, point in self.goc_mpc.last_grasp_commands:
        #         if robot == "free_body_0" or robot == "point_mass_0":
        #             side = "left"
        #         elif robot == "free_body_1" or robot == "point_mass_1":
        #             side = "right"
        #         else:
        #             continue
        #         self.get_logger().info(f"Paused {side}!")
        #         self._pause_robot_delayed(
        #             side=side,
        #             pre_delay=self._grasp_settle_sec,
        #             post_delay=self._grasp_pause_after_cmd_sec,
        #             gripper_cmd=cmd
        #         )

        # if len(self.goc_mpc.last_cycle_backtracked_phases) > 0:
        #     for agent_idx, new_phase in self.goc_mpc.last_cycle_backtracked_phases.items():
        #         if agent_idx == 0:
        #             side = "left"
        #         elif agent_idx == 1:
        #             side = "right"
        #         else:
        #             continue
        #         self.get_logger().info(f"Paused {side} to backtrack!")
        #         self._pause_robot_delayed(
        #             side=side,
        #             pre_delay=0.0,
        #             post_delay=0.0,
        #             gripper_cmd="release"
        #         )

        if not self._robot_paused and target_pose_stamped is not None:
            self.target_pose_publisher.publish(target_pose_stamped)

            next_gripper = float(gripper_target[0])
            prev = self._last_commanded_gripper
            if next_gripper > 0.0 and prev <= 0.0:
                self.get_logger().info("Closing gripper.")
                self._last_commanded_gripper = 1.0
                self._pause_robot_delayed(
                    pre_delay=self._grasp_settle_sec,
                    post_delay=self._grasp_pause_after_cmd_sec,
                    gripper_cmd="close",
                )
            elif next_gripper <= 0.0 and prev > 0.0:
                self.get_logger().info("Opening gripper.")
                self._last_commanded_gripper = -1.0
                self._pause_robot_delayed(
                    pre_delay=0.0,
                    post_delay=0.0,
                    gripper_cmd="open",
                )

    # --- Helpers ---

    def _quat_to_yaw(self, quat) -> float:
        """Extract yaw (rotation about world Z) from a geometry_msgs Quaternion."""
        _, _, yaw = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
        return yaw

    def _yaw_to_quat(self, yaw: float):
        """Convert yaw angle to (w, x, y, z) with the fixed roll/pitch used by the robot."""
        qx, qy, qz, qw = quaternion_from_euler(-np.pi, 0.0, yaw)
        return (qw, qx, qy, qz)

    def _publish_paths(self, left_path_pub, left_xi, right_path_pub, right_xi, pos_only=True):
        left_path_msg = Path()
        left_path_msg.header.frame_id = WORLD_FRAME   # or "map", depending on your TF setup
        left_path_msg.header.stamp = self.get_clock().now().to_msg()

        for row in left_xi:
            pose = PoseStamped()
            pose.header = left_path_msg.header
            if pos_only:
                x, y, z = row[:3]
                qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
            else:
                # take the first 7 elements of the row (first pose)
                x, y, z, qw, qx, qy, qz = row[:7]
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = float(z)
            pose.pose.orientation.w = float(qw)
            pose.pose.orientation.x = float(qx)
            pose.pose.orientation.y = float(qy)
            pose.pose.orientation.z = float(qz)
            left_path_msg.poses.append(pose)

        left_path_pub.publish(left_path_msg)

        right_path_msg = Path()
        right_path_msg.header.frame_id = WORLD_FRAME   # or "map", depending on your TF setup
        right_path_msg.header.stamp = self.get_clock().now().to_msg()

        for row in right_xi:
            pose = PoseStamped()
            pose.header = right_path_msg.header
            if pos_only:
                x, y, z = row[:3]
                qw, qx, qy, qz = 0.0, 0.0, 1.0, 0.0
            else:
                # take the first 7 elements of the row (first pose)
                x, y, z, qw, qx, qy, qz = row[:7]
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = float(z)
            pose.pose.orientation.w = float(qw)
            pose.pose.orientation.x = float(qx)
            pose.pose.orientation.y = float(qy)
            pose.pose.orientation.z = float(qz)
            right_path_msg.poses.append(pose)

        right_path_pub.publish(right_path_msg)


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

    def _resume_robot(self):
        self._robot_paused = False
        if self._resume_timer is not None:
            self._resume_timer.cancel()
            self._resume_timer = None
        self.get_logger().info("robot resumed after grasp pause.")

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
        self._resume_timer = self.create_timer(self._grasp_pause_after_cmd_sec,
                                               self._resume_robot)

    def _pause_robot_delayed(self, pre_delay: float, post_delay: float, gripper_cmd: str):
        """
        Immediately pause robot, wait pre_delay, then execute gripper_cmd, then
        wait post_delay and resume. If re-triggered, refresh the sequence.
        """
        self._robot_paused = True
        self._pending_gripper_cmd = gripper_cmd

        if self._pre_grasp_timer is not None:
            self._pre_grasp_timer.cancel()
            self._pre_grasp_timer = None
        self._pre_grasp_timer = self.create_timer(pre_delay, self._on_pre_grasp)

        if self._resume_timer is not None:
            self._resume_timer.cancel()
            self._resume_timer = None

    def _to_world(self, pose_msg: PoseStamped, timeout_sec: float = 0.1, target_frame: str = WORLD_FRAME) -> Optional[PoseStamped]:
        """Turn a PoseStamped (using its header.frame_id) into a PoseStamped in the target frame."""
        if pose_msg is None:
            return None
        src_frame = pose_msg.header.frame_id
        if not src_frame:
            self.get_logger().warn("Incoming PoseStamped has empty header.frame_id")
            return None
        if src_frame == target_frame:
            return pose_msg  # already in target_frame

        try:
            # Get transform: target <- source (i.e., world <- src_frame)
            tf: 'TransformStamped' = self.tf_buffer.lookup_transform(
                target_frame,               # target frame
                src_frame,                  # source frame
                Time(), # pose_msg.header.stamp,      # Time(), # use the pose time if timestamps are reasonable
                timeout=rclpy.duration.Duration(seconds=timeout_sec)
            )
            pose_stamped_world: PoseStamped = do_transform_pose_stamped(pose_msg, tf)
            # pose_world.header.frame_id = WORLD_FRAME  # make sure it says 'world'
            # keep the original timestamp (or set to now() if you prefer)
            return pose_stamped_world
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(
                f"TF transform failed ({WORLD_FRAME} <- {src_frame}) at t={pose_msg.header.stamp.sec}.{pose_msg.header.stamp.nanosec}: {e}"
            )
            return None


    def _twist_to_world(self, twist_msg: TwistStamped, timeout_sec: float = 0.05) -> Optional[Twist]:
        """Turn a TwistStamped (using its header.frame_id) into a Twist in WORLD_FRAME."""
        if twist_msg is None:
            return None
        src_frame = twist_msg.header.frame_id
        if not src_frame:
            self.get_logger().warn("Incoming PoseStamped has empty header.frame_id")
            return None
        if src_frame == WORLD_FRAME:
            return twist_msg.twist  # already in world

        try:
            tf = self.tf_buffer.lookup_transform(
                WORLD_FRAME,
                src_frame,
                Time(), # twist_msg.header.stamp,
                timeout=rclpy.duration.Duration(seconds=timeout_sec),
            )

            p = tf.transform.translation
            q = tf.transform.rotation
            R = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]

            skew_symmetric_p =  np.array([
                [ 0,   -p.z,  p.y],
                [ p.z,  0,   -p.x],
                [-p.y,  p.x,  0]
            ])

            adjoint_T_ab = np.concatenate([
                np.concatenate([R, np.zeros((3,3))], axis=1),
                np.concatenate([np.matmul(skew_symmetric_p, R), R], axis=1)
            ], axis=0)


            twist_b = np.array([
                twist_msg.twist.linear.x,
                twist_msg.twist.linear.y,
                twist_msg.twist.linear.z,
                twist_msg.twist.angular.x,
                twist_msg.twist.angular.y,
                twist_msg.twist.angular.z,
            ])

            twist_a = np.matmul(adjoint_T_ab, twist_b)

            twist_world = Twist()
            twist_world.linear.x = twist_a[0]
            twist_world.linear.y = twist_a[1]
            twist_world.linear.z = twist_a[2]
            twist_world.angular.x = twist_a[3]
            twist_world.angular.y = twist_a[4]
            twist_world.angular.z = twist_a[5]
            return twist_world
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(
                f"TF transform failed ({WORLD_FRAME} <- {src_frame}) at "
                f"t={twist_msg.header.stamp.sec}.{twist_msg.header.stamp.nanosec}: {e}"
            )
            return None


def main(args=None):
    rclpy.init(args=args)

    node = OneRobotGCRLManipulationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
