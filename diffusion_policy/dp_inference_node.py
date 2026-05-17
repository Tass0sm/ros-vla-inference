#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

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
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty

from tf2_ros import (
    Buffer,
    TransformListener,
    LookupException,
    ConnectivityException,
    ExtrapolationException,
)
from tf2_geometry_msgs import do_transform_pose_stamped

from robomimic.utils import file_utils as FileUtils
from robomimic.utils import obs_utils as ObsUtils
from robomimic.utils import python_utils as PyUtils

_GUIDED_DIFFUSION_ROOT = Path("/home/moritz/src/guided-diffusion")
if str(_GUIDED_DIFFUSION_ROOT) not in sys.path:
    sys.path.insert(0, str(_GUIDED_DIFFUSION_ROOT))

try:
    from calvin_experiments.calvin_rollout_utils import automaton_model_for_eval as _load_automaton
    _AUTOMATON_LOADER_AVAILABLE = True
except ImportError:
    _AUTOMATON_LOADER_AVAILABLE = False

from goc_demo import robotiq


WORLD_FRAME = "world"

POUR_TWIST_ON_RAD  = np.deg2rad(90.0)  # wrist relative-angle threshold to enter pouring
POUR_TWIST_OFF_RAD = np.deg2rad(75.0)  # hysteresis off-threshold (unused in single-step check)

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
        self.declare_parameter("ckpt_path", "/home/shared/models/real_world/cheezit_dp_20260513170400/model_epoch_160.pth")
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("gripper_mid", 50)  # raw gripper threshold: below→open(-1), above→closed(+1)
        self.declare_parameter("preview_chunks", False)  # pause after each inference until /resume is published
        self.declare_parameter("n_candidates", 1)  # number of action chunks to sample per inference step
        self.declare_parameter(
            "automaton_ckpt_path",
            "/home/moritz/src/guided-diffusion/outputs/real_world/automaton_world_model"
            "/h8_max_next_lr0.0001_epochs80_2026-05-13_21-39-13/best_model.pt",
        )
        self.declare_parameter("target_label_idx", 0)
        self.declare_parameter("target_chain", "")  # "idx:maximize;idx:minimize;..."
        self.declare_parameter("wrist_joint_topic", "/joint_states")
        self.declare_parameter("wrist_joint_name", "wrist_3_joint")

        self._pose_topic: str = self.get_parameter("pose_topic").value
        self._twist_topic: str = self.get_parameter("twist_topic").value
        self._obj_pose_topic: str = self.get_parameter("obj_pose_topic").value
        self._ckpt_path: str = self.get_parameter("ckpt_path").value
        self._rate_hz: float = float(self.get_parameter("rate_hz").value)
        self._gripper_mid: float = float(self.get_parameter("gripper_mid").value)
        self._preview_chunks: bool = bool(self.get_parameter("preview_chunks").value)
        self._n_candidates: int = max(1, int(self.get_parameter("n_candidates").value))
        self._automaton_ckpt_path: str = self.get_parameter("automaton_ckpt_path").value
        self._target_label_idx: int = int(self.get_parameter("target_label_idx").value)
        self._wrist_joint_topic: str = self.get_parameter("wrist_joint_topic").value
        self._wrist_joint_name: str = self.get_parameter("wrist_joint_name").value
        try:
            _chain_raw = self.get_parameter("target_chain").value or ""
            self._target_chain: list = [
                (int(p.split(":")[0].strip()), p.split(":")[1].strip())
                for p in _chain_raw.split(";") if p.strip()
            ]
        except Exception as e:
            self.get_logger().error(f"Failed to parse target_chain '{_chain_raw}': {e}")
            self._target_chain = []
        self._chain_pos: int = 0
        self._chain_done: bool = False

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

        # --- Action queue (our own chunk buffer, populated by _refill_action_queue) ---
        self._action_queue: list = []
        self._awaiting_chunk_resume: bool = False  # True when paused for preview

        if self._preview_chunks:
            self.create_subscription(Empty, "~/resume", self._on_resume, 10)
            self.get_logger().info(
                "Chunk preview mode enabled. "
                "Publish to ~/resume to execute each new chunk."
            )

        # --- Observation history (frame stack = OBS_HORIZON) ---
        self._obs_deque: deque = deque(maxlen=self.OBS_HORIZON)
        self._gripper_raw_history: deque = deque(maxlen=2)
        self._last_commanded_gripper: float = -1.0  # track last commanded state to detect transitions

        # --- Wrist joint for pouring labels ---
        self._latest_wrist_rad: Optional[float] = None
        self._start_wrist_rad: Optional[float] = None  # set on first joint state; reference for relative twist
        self.create_subscription(JointState, self._wrist_joint_topic, self._on_joint_states, best_effort_qos)

        # --- Load diffusion policy ---
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self.get_logger().info(f"Loading checkpoint: {self._ckpt_path} on {device}")
        self._policy, _ = FileUtils.policy_from_checkpoint(
            ckpt_path=self._ckpt_path,
            device=device,
            verbose=True,
        )
        self._policy.start_episode()
        self.get_logger().info("Diffusion policy loaded.")

        # --- Load automaton world model ---
        self._automaton_model = None
        self._automaton_stats = None
        self._automaton_meta = None
        if self._automaton_ckpt_path and _AUTOMATON_LOADER_AVAILABLE:
            try:
                self._automaton_model, self._automaton_stats, self._automaton_meta = \
                    _load_automaton(Path(self._automaton_ckpt_path), device)
                label_names = self._automaton_meta["label_names"]
                self.get_logger().info(
                    f"Automaton loaded: {self._automaton_ckpt_path}\n"
                    f"  labels={label_names}  "
                    f"target=[{self._target_label_idx}] {label_names[self._target_label_idx]}"
                )
                if self._target_chain:
                    chain_desc = ", ".join(
                        f"{mode}({label_names[idx]})" for idx, mode in self._target_chain
                    )
                    self.get_logger().info(f"Target chain: [{chain_desc}]")
            except Exception as e:
                self.get_logger().error(f"Failed to load automaton model: {e}")
        elif not _AUTOMATON_LOADER_AVAILABLE:
            self.get_logger().warn("automaton_model_for_eval not importable; no automaton guidance.")

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

    def _on_joint_states(self, msg: JointState):
        try:
            idx = msg.name.index(self._wrist_joint_name)
        except ValueError:
            return
        pos = float(msg.position[idx])
        if self._start_wrist_rad is None:
            self._start_wrist_rad = pos
            self.get_logger().info(f"Wrist reference set: {self._wrist_joint_name} = {pos:.4f} rad")
        self._latest_wrist_rad = pos

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
        self._gripper_raw_history.append(gripper_raw)

        eef_pos = np.array([pose.position.x,
                             pose.position.y,
                             pose.position.z], dtype=np.float32)
        eef_rot6d = _quat_wxyz_to_rot6d(_ros_quat_to_wxyz(pose.orientation)).astype(np.float32)

        cheezit_pos = np.array([obj_pose.position.x,
                                 obj_pose.position.y,
                                 obj_pose.position.z], dtype=np.float32)
        cheezit_rot6d = _quat_wxyz_to_rot6d(_ros_quat_to_wxyz(obj_pose.orientation)).astype(np.float32)

        step_obs = {
            "eef_pos":        eef_pos,
            "eef_rot6d":      eef_rot6d,
            "gripper_binary": np.array([gripper_binary], dtype=np.float32),
            "cheezit_pos":    cheezit_pos,
            "cheezit_rot6d":  cheezit_rot6d,
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
        #  Advance target chain based on current estimated label             #
        # ------------------------------------------------------------------ #

        if self._target_chain and self._automaton_model is not None and not self._chain_done:
            _, current_label = self._current_automaton_state_label(step_obs)
            if self._check_and_advance_chain(current_label):
                self._action_queue.clear()
        if self._chain_done:
            return

        if not self._robot_paused and not self._awaiting_chunk_resume:

            # ------------------------------------------------------------------ #
            #  Refill action queue when empty, then pop one action               #
            # ------------------------------------------------------------------ #

            if not self._action_queue:
                self._refill_action_queue(stacked_obs, step_obs)
                if self._awaiting_chunk_resume or not self._action_queue:
                    return

            action = self._action_queue.pop(0)  # numpy array, shape (7,)

            # action: [dx, dy, dz, drotvec_x, drotvec_y, drotvec_z, next_gripper_binary]
            dpos     = action[:3].astype(np.float64)
            drotvec  = action[3:6].astype(np.float64)
            next_gripper = float(action[6])

            # ------------------------------------------------------------------ #
            #  Compute next EEF pose                                              #
            # ------------------------------------------------------------------ #

            new_pos = np.array([pose.position.x,
                                 pose.position.y,
                                 pose.position.z], dtype=np.float64) + dpos

            # q_new = q_delta * q_current  (convention from training)
            q_cur_wxyz   = _ros_quat_to_wxyz(pose.orientation)
            q_delta_wxyz = _rotvec_to_quat_wxyz(drotvec)
            q_new_wxyz   = _quat_wxyz_multiply(q_delta_wxyz, q_cur_wxyz)

            # ------------------------------------------------------------------ #
            #  Publish target pose                                                #
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

            self.target_pose_publisher.publish(target)

            # ------------------------------------------------------------------ #
            #  Execute gripper command on state transition                        #
            # ------------------------------------------------------------------ #

            prev = self._last_commanded_gripper
            if next_gripper > 0.0 and prev <= 0.0:
                self.get_logger().info("closing gripper!")
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
    #  Action queue helpers                                                    #
    # ---------------------------------------------------------------------- #

    def _refill_action_queue(self, stacked_obs: dict, step_obs: dict):
        """Sample n_candidates action chunks, score them, fill _action_queue with the best."""
        try:
            obs_tensor = self._policy._prepare_observation(stacked_obs)
            obs_batch = self._repeat_obs_batch(obs_tensor, self._n_candidates)
            with torch.no_grad():
                action_chunks_t = self._policy.policy._get_action_trajectory(obs_dict=obs_batch)
            action_chunks = self._unnormalize_action_chunks(action_chunks_t)  # (N, H, A)
        except Exception as e:
            self.get_logger().error(f"Policy inference failed: {e}")
            return

        if self._automaton_model is not None:
            automaton_state, automaton_label = self._current_automaton_state_label(step_obs)
            label_probs = self._predict_future_label_probs(automaton_state, automaton_label, action_chunks)
            if self._target_chain and not self._chain_done:
                idx, mode = self._target_chain[self._chain_pos]
                scores = label_probs[:, idx]
                selected_idx = int(np.argmax(scores)) if mode == "maximize" else int(np.argmin(scores))
                score_label = f"{'max' if mode == 'maximize' else 'min'} p({self._automaton_meta['label_names'][idx]})"
            else:
                scores = label_probs[:, self._target_label_idx]
                selected_idx = int(np.argmax(scores))
                score_label = f"p({self._automaton_meta['label_names'][self._target_label_idx]})"
        else:
            selected_idx, scores = self._score_candidates(action_chunks)
            score_label = "sum_dx"

        # _obs_lines = "\n".join(
        #     f"  {k}:\n{np.array2string(v, precision=4, floatmode='fixed', separator=', ')}"
        #     for k, v in stacked_obs.items()
        # )
        # _gripper_hist = "  ".join(f"{v:.4f}" for v in self._gripper_raw_history)
        # self.get_logger().info(f"\n\nstacked_obs:\n{_obs_lines}\n  gripper_raw (last 2): {_gripper_hist}\n\n")

        self._action_queue = list(action_chunks[selected_idx])  # list of H arrays, each shape (A,)

        if self._automaton_model is not None:
            label_names = self._automaton_meta["label_names"]
            col_w = max(max(len(n) for n in label_names), 6)
            header = f"  {'':3s}  " + "  ".join(f"{n:>{col_w}}" for n in label_names)
            rows = []
            for i, probs in enumerate(label_probs):
                marker = "* " if i == selected_idx else "  "
                rows.append(f"  {marker}{i:3d}  " + "  ".join(f"{p:{col_w}.4f}" for p in probs))
            current_label_line = "  current label:  " + "  ".join(
                f"{float(automaton_label[j]):{col_w}.4f}" for j in range(len(label_names))
            )
            self.get_logger().info(
                f"World model predictions [{score_label}] "
                f"(N={self._n_candidates}, selected={selected_idx}):\n"
                + header + "\n" + "\n".join(rows) + "\n" + current_label_line
            )

        if self._preview_chunks:
            self._awaiting_chunk_resume = True
            self.get_logger().info("Chunk preview: publish to ~/resume to begin executing.")

    @staticmethod
    def _repeat_obs_batch(obs_tensor: dict, n: int) -> dict:
        """Tile a batch-1 obs tensor to batch-n by repeating along the first dimension."""
        if n == 1:
            return obs_tensor
        return {k: v.repeat((n,) + (1,) * (v.ndim - 1)) for k, v in obs_tensor.items()}

    def _unnormalize_action_chunks(self, action_tensor) -> np.ndarray:
        """Unnormalize action tensor (N, H, A) → numpy (N, H, A)."""
        action_np = action_tensor.detach().cpu().numpy().astype(np.float32)  # (N, H, A)
        if self._policy.action_normalization_stats is None:
            return action_np
        original_shape = action_np.shape
        flat = action_np.reshape(-1, original_shape[-1])  # (N*H, A)
        action_keys = self._policy.policy.global_config.train.action_keys
        action_shapes = {
            key: self._policy.action_normalization_stats[key]["offset"].shape[1:]
            for key in self._policy.action_normalization_stats
        }
        action_dict = PyUtils.vector_to_action_dict(flat, action_shapes=action_shapes, action_keys=action_keys)
        action_dict = ObsUtils.unnormalize_dict(action_dict, normalization_stats=self._policy.action_normalization_stats)
        return PyUtils.action_dict_to_vector(action_dict, action_keys=action_keys).reshape(original_shape)

    def _score_candidates(self, action_chunks: np.ndarray) -> tuple:
        """Fallback scorer: min sum of x-translation deltas. Returns (argmin_idx, scores)."""
        scores = action_chunks[:, :, 0].sum(axis=1)  # (N,) — action index 0 = dx
        return int(np.argmin(scores)), scores

    def _check_and_advance_chain(self, current_label: np.ndarray) -> bool:
        """Advance chain_pos while the current target is satisfied. Returns True if advanced."""
        advanced = False
        while self._chain_pos < len(self._target_chain):
            idx, mode = self._target_chain[self._chain_pos]
            label_val = float(current_label[idx])
            satisfied = (label_val > 0.5) if mode == "maximize" else (label_val < 0.5)
            if not satisfied:
                break
            name = self._automaton_meta["label_names"][idx]
            self.get_logger().info(
                f"Chain [{self._chain_pos + 1}/{len(self._target_chain)}] "
                f"{mode}({name}) satisfied (label={label_val:.3f}) — advancing"
            )
            self._chain_pos += 1
            advanced = True
        if self._chain_pos >= len(self._target_chain):
            self._chain_done = True
            self.get_logger().info("finished")
        return advanced

    def _current_automaton_state_label(self, step_obs: dict) -> tuple:
        """Build the 19-dim automaton state and estimate the current label from step_obs."""
        # Key order matches real_world_data.py state_keys
        state = np.concatenate([
            step_obs["eef_pos"],        # 3
            step_obs["eef_rot6d"],      # 6
            step_obs["gripper_binary"], # 1
            step_obs["cheezit_pos"],    # 3
            step_obs["cheezit_rot6d"],  # 6
        ]).astype(np.float32)           # total: 19

        label_dim = len(self._automaton_meta["label_names"])
        label = np.zeros(label_dim, dtype=np.float32)

        gripper_closed = step_obs["gripper_binary"][0] > 0.0
        label[0] = 1.0 if gripper_closed else 0.0

        # pouring labels: relative wrist_3 twist from episode-start reference
        if self._latest_wrist_rad is not None and self._start_wrist_rad is not None:
            twist = self._latest_wrist_rad - self._start_wrist_rad
            label[1] = 1.0 if (gripper_closed and twist >  POUR_TWIST_ON_RAD) else 0.0  # pouring_right
            label[2] = 1.0 if (gripper_closed and twist < -POUR_TWIST_ON_RAD) else 0.0  # pouring_left

        return state, label

    def _predict_future_label_probs(
        self,
        automaton_state: np.ndarray,
        automaton_label: np.ndarray,
        action_chunks: np.ndarray,
    ) -> np.ndarray:
        """Run the automaton model; return predicted label probs of shape (N, label_dim)."""
        n_candidates, _, action_dim = action_chunks.shape
        action_chunk_dim = len(self._automaton_stats["actions_mean"])
        automaton_horizon = action_chunk_dim // action_dim

        # Flatten only the automaton-horizon steps; ignore any extra policy steps
        flat_chunks = action_chunks[:, :automaton_horizon, :].reshape(n_candidates, -1)

        states = np.repeat(automaton_state[None, :], n_candidates, axis=0)
        labels = np.repeat(automaton_label[None, :], n_candidates, axis=0)

        states_t = torch.as_tensor(
            (states - self._automaton_stats["states_mean"]) / self._automaton_stats["states_std"],
            device=self._device, dtype=torch.float32,
        )
        actions_t = torch.as_tensor(
            (flat_chunks - self._automaton_stats["actions_mean"]) / self._automaton_stats["actions_std"],
            device=self._device, dtype=torch.float32,
        )
        labels_t = torch.as_tensor(labels, device=self._device, dtype=torch.float32)

        with torch.no_grad():
            logits = self._automaton_model(states_t, actions_t, labels_t)
            return torch.sigmoid(logits).detach().cpu().numpy()  # (N, label_dim)

    def _on_resume(self, _msg: Empty):
        if self._awaiting_chunk_resume:
            self._awaiting_chunk_resume = False
            self.get_logger().info(f"Resuming execution of {len(self._action_queue)}-step chunk.")

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
