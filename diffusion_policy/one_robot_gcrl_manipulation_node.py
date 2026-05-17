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
from tf_transformations import quaternion_matrix

from pydrake.math import RollPitchYaw
from pydrake.common.eigen_geometry import Quaternion

# from goc_mpc.splines import Block
# from goc_mpc.goc_mpc import GraphOfConstraints, GraphOfConstraintsMPC
# from goc_mpc.simple_drake_env import SimpleDrakeGym

import jax

from agents.pixel_agents.h_flow_hjb_gcivl import H_Flow_HJB_GCIVL_PixelAgent
from utils.flax_utils import restore_agent
from utils.datasets import Dataset, PixelHGCDataset

from goc_demo import robotiq


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

        self.bridge = CvBridge()
        self._target_img_dim = 128

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
        self._obs_deque = deque(maxlen=3)
        self.create_subscription(Image, '/camera/camera/color/image_raw', self._on_image, 10)

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

        # Tunables
        self._grasp_settle_sec = 1.00          # wait before actuating gripper
        self._grasp_pause_after_cmd_sec = 1.00 # time to remain paused after actuation

        # Initialize agent.
        config = dict(
            # Agent hyperparameters.
            agent_name='h_flow_hjb_gcivl_pixel',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(1024, 1024, 1024, 1024),  # Actor network hidden dimensions.
            value_hidden_dims=(1024, 1024, 1024, 1024),  # Value network hidden dimensions.
            expectile=0.9,  # IQL expectile.
            alpha=10.0,  # Temperature in AWR or BC coefficient.
            layer_norm=True,  # Whether to use layer normalization.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            discount=0.999,  # Discount factor.
            # low_discount=ml_collections.config_dict.placeholder(float),  # Low-level discount (set automatically).
            tau=0.005,  # Target network update rate.
            q_agg='min',  # Aggregation function for Q values.
            # action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (set automatically).
            # goal_dim=ml_collections.config_dict.placeholder(tuple),  # Goal dimension (set automatically).
            # subgoal_dim=ml_collections.config_dict.placeholder(int),  # Goal dimension (set automatically).
            value_loss_type='bce',  # Value loss type ('squared' or 'bce').
            flow_steps=10,  # Number of flow steps.
            num_samples=32,  # Number of samples for the actor.
            encoder="impala_small",  # Visual encoder name (None, 'impala_small', etc.).
            # Dataset hyperparameters.
            dataset_class='PixelHGCDataset',  # Dataset class name.
            subgoal_steps=10,  # Subgoal steps.
            value_p_curgoal=0.2,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.3,  # Probability of using a random state as the value goal.
            value_geom_sample=False,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=True,  # Whether to use geometric sampling for future actor goals.
            gc_negative=False,  # Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as reward.
            p_aug=0.5,  # Probability of applying image augmentation.
            frame_stack=3,  # Number of frames to stack.
        )

        with open("/home/tassos/phd/software/ros_workspaces/test_ws/src/diffusion_policy/data/processed_dataset_v3.pkl", "rb") as f:
            train_dataset = pickle.load(f)

        self._action_min = train_dataset.pop("action_min", None)
        self._action_max = train_dataset.pop("action_max", None)
        self._action_norm = train_dataset.pop("actions_norm", None)

        train_dataset['terminals'][-1] = 1.0

        train_dataset = PixelHGCDataset(Dataset.create(**train_dataset), config)

        goal_index = np.argwhere(train_dataset.dataset["terminals"])[0].item()
        self._goal_obs = np.expand_dims(train_dataset.dataset["observations"][goal_index], 0)

        example_batch = train_dataset.sample(1)

        agent = H_Flow_HJB_GCIVL_PixelAgent.create(
            0,
            example_batch,
            config,
            "real_world_experiment",
        )
        self.agent = restore_agent(
            agent,
            "/home/tassos/phd/software/ros_workspaces/test_ws/src/real_world_checkpoints/h_flow_hjb_gcivl/version3",
            "200000"
        )

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

        # Only using cartesian position
        def pose_to_arr(pose: Pose):
            return np.array([pose.position.x,
                             pose.position.y,
                             pose.position.z])

        # def twist_to_arr(twist: Twist):
        #     return np.array([twist.linear.x,
        #                      twist.linear.y,
        #                      twist.linear.z])

        x = np.concatenate([q, # qd, eff,
                            pose_to_arr(pose),
                            # np.array([0.0, 0.0, 1.0, 0.0]),
                            # twist_to_arr(twist),
                            np.array([gripper_pos], dtype=float)])

        return x

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

        if self._latest_image is None:
            self.get_logger().info('_latest_image is None')
            return

        now = self.get_clock().now()
        t = (now - self._start_time).nanoseconds * 1e-9

        #######################################################################
        #                           GET OBSERVATION                           #
        #######################################################################


        try:
            self._obs_deque.append(self._latest_image)
            assert len(self._obs_deque) == 3, "Need three latest observations. Skipping."
            observation = np.concatenate(self._obs_deque, axis=-1)
            proprioception = self._extract_state(self._latest_q,
                                                 # self._latest_qd,
                                                 # self._latest_eff,
                                                 self._latest_pose,
                                                 # self._latest_twist,
                                                 self._real_gripper.get_current_position())
        except Exception as e:
            self.get_logger().warn(f"Bad State: {e}")
            return

        #######################################################################
        #                             AGENT STEP                              #
        #######################################################################

        action_norm = self.agent.sample_actions(observation, proprioception,
                                                goals=self._goal_obs, seed=jax.random.PRNGKey(0))
        action_norm = np.asarray(action_norm)
        delta_norm = action_norm[:3]
        delta = self._denormalize_actions(delta_norm)

        # action is left pose delta in world frame. latest_left_pose is in world frame
        target = delta + np.array([self._latest_pose.position.x,
                                   self._latest_pose.position.y,
                                   self._latest_pose.position.z])
        gripper_target = action_norm[3:] # just [-1.0] or [1.0]

        self.get_logger().info(f"delta: {delta}, gripper: {gripper_target}")

        action = np.concatenate((target, gripper_target), axis=-1)

        #######################################################################
        #                            EXECUTE ACTION                           #
        #######################################################################

        target_pose_stamped = PoseStamped()
        target_pose_stamped.header.frame_id = WORLD_FRAME
        target_pose_stamped.header.stamp = self.get_clock().now().to_msg()
        target_pose_stamped.pose.position.x = action[0]
        target_pose_stamped.pose.position.y = action[1]
        target_pose_stamped.pose.position.z = action[2]
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

    # --- Helpers ---

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


    def _do_gripper_cmd(self, side: str, cmd: str):
        try:
            gr = self.left_real_gripper if side == 'left' else self.right_real_gripper
            if cmd == 'grab':
                gr.close(speed=200, force=2)
            elif cmd == 'release':
                gr.open(speed=200, force=2)
            else:
                self.get_logger().warn(f"Unknown gripper cmd: {cmd}")
        except Exception as e:
            self.get_logger().error(f"Gripper {side} command '{cmd}' failed: {e}")

    def _resume_robot(self):
        self._robot_paused = False
        if self._resume_timer is not None:
            self._resume_timer.cancel()
            self._resume_timer = None
        self.get_logger().info("robot resumed after grasp pause.")

    def _on_left_pre_grasp(self):
        """Fires after settle delay: actuate gripper then start resume timer."""
        if self._left_pre_grasp_timer is not None:
            self._left_pre_grasp_timer.cancel()
            self._left_pre_grasp_timer = None
        cmd = self._left_pending_gripper_cmd
        self._left_pending_gripper_cmd = None
        if cmd is not None:
            self._do_gripper_cmd('left', cmd)
        # chain the resume one-shot
        if self._left_resume_timer is not None:
            self._left_resume_timer.cancel()
            self._left_resume_timer = None
        self._left_resume_timer = self.create_timer(self._grasp_pause_after_cmd_sec,
                                                    self._resume_robot_left)

    def _on_pre_grasp(self):
        if self._pre_grasp_timer is not None:
            self._pre_grasp_timer.cancel()
            self._pre_grasp_timer = None
        cmd = self._pending_gripper_cmd
        self._pending_gripper_cmd = None
        if cmd is not None:
            self._do_gripper_cmd('right', cmd)
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
