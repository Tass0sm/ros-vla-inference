#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 21:13:15 2026

@author: exx
"""

from pathlib import Path
import pickle
import numpy as np
from PIL import Image

# ============================================================
# CONFIG
# ============================================================

input_folder = Path("/home/tassos/phd/software/ros_workspaces/test_ws/saved_data/pick_and_place_spam")
output_file = Path("processed_dataset_state_based_v12.pkl")
# output_file = Path("processed_dataset_image_based_v5.pkl")

# "image"  — observations are camera frames (H×W×C)
# "proprio" — observations are the proprioception vector (no images loaded)
OBS_MODE = "proprio"

# Only used when OBS_MODE == "image"
RESIZE_IMAGES = True
IMAGE_SIZE    = 64       # target square resolution (pixels)

# Include spam (x, y, z, yaw) in the proprioception vector
INCLUDE_SPAM_POSE = True

# Action source — four options (set exactly one to True):
#
#   USE_EE_DELTA_ACTIONS = True   — use ee_pos[t+1]-ee_pos[t] everywhere.
#       Naturally near-zero during pauses but is ~10× smaller in magnitude
#       than the MPC commands.  Causes micro-movement policy at deploy time.
#
#   ZERO_PAUSE_ACTIONS = True     — use MPC actions everywhere, but replace
#       the pause windows with zeros.  Correct scale but the policy learns
#       "do-nothing" labels at the most critical moments (~20% of steps).
#
#   SPLICE_PAUSE_WITH_EE = True   — use MPC actions everywhere EXCEPT the
#       pause windows, where EE deltas replace them.  Removes the upward
#       artefact while preserving MPC scale, but introduces discontinuities
#       at every splice boundary that are visible as action jerk spikes.
#
#   SCALE_EE_TO_MPC = True        — use EE deltas everywhere, but scale each
#       dimension so its std matches the corresponding MPC action std.
#       Preserves the smooth temporal profile (near-zero during pauses, no
#       upward kick) while deploying at the correct controller scale.
#       Recommended.
USE_EE_DELTA_ACTIONS    = False
ZERO_PAUSE_ACTIONS      = False
SPLICE_PAUSE_WITH_EE    = False
SCALE_EE_TO_MPC         = True

# If True, relabel next_gripper_pos for every step inside a gripper pause
# window so that it always predicts the POST-transition state.
#
# Without this, the pause window (settle + resume ≈ 20 steps) has only one
# step that carries the "transition" gripper label (the step immediately
# before the physical transition), while all other window steps say "hold
# current state" — a 19:1 imbalance that teaches the policy never to
# change gripper state.  Relabeling gives the policy a consistent signal:
# "whenever you are in this window, initiate the gripper change."
# Compatible with all four action modes above.
RELABEL_PAUSE_GRIPPER = True

# Controls how far *before* the physical gripper transition the relabeling
# reaches back into the settle window.
#
#   0.0 — resume-only (default): relabel starts exactly at the transition
#         step; the settle window keeps its natural (hold-current) label.
#         Prevents the policy from triggering the gripper early.
#
#   1.0 — full window: relabel covers the entire settle window too
#         (original v9 behaviour).  Fixes the 19:1 imbalance fully but
#         causes the policy to fire ~GRASP_SETTLE_SEC too early.
#
#   0.5 — halfway: relabel starts halfway through the settle window.
#         A reasonable compromise if the policy still misses some grasps
#         with 0.0.
#
# Has no effect when RELABEL_PAUSE_GRIPPER = False.
RELABEL_SETTLE_FRACTION = 0.3

# If True, remove every step inside a gripper pause window from the dataset
# entirely.  The robot is physically stationary during these windows (EE does
# not move), so with SCALE_EE_TO_MPC the actions are near-zero for
# settle_steps + resume_steps ≈ 20 consecutive steps at the exact grasp /
# release location.  The policy over-learns "stay here" and gets stuck.
#
# Trimming removes those steps so the dataset jumps directly from the last
# approaching step to the first departing step.  The controller still handles
# the physical gripper close/open timing; the policy does not need to dwell.
#
# When used together with RELABEL_PAUSE_GRIPPER (recommended), the relabeling
# acts on the trimmed sequence: the transition appears as a single-step jump,
# and RELABEL_SETTLE_FRACTION controls how many approach steps before the jump
# are relabeled to "initiate gripper change" (e.g. 0.3 → 3 steps).
TRIM_PAUSE_WINDOWS = False

RATE_HZ            = 10.0          # recording rate used during collection
GRASP_SETTLE_SEC   = 1.0           # _grasp_settle_sec in the demo node
GRASP_RESUME_SEC   = 1.0           # _grasp_pause_after_cmd_sec in the demo node
GRIPPER_OPEN_MAX   = 20            # gripper_pos values ≤ this are "open"

# Gripper normalization for proprioception.
# The raw gripper_pos (an integer 0–~200) is mapped to [-1, +1] before being
# stored in the observation vector, consistent with the ±1 action convention
# (+1 = closed, -1 = open).  Values outside [MIN, MAX] are clipped to ±1.
# Only affects the observation; all internal logic (pause masks, action labels)
# still uses the raw value.
#
# NOTE: the ROS deployment node must apply the same normalization when
# constructing observations so the policy sees the same scale it trained on.
GRIPPER_PROP_MIN   = 0.0           # raw gripper_pos mapped to -1 (fully open)
GRIPPER_PROP_MAX   = 200.0         # raw gripper_pos mapped to +1 (fully closed)

# ============================================================
# HELPERS
# ============================================================

def compute_ee_delta_actions(data: dict, keep: np.ndarray) -> np.ndarray:
    """Derive actions from the actual observed EE displacement between steps.

    Returns ee_pos[t+1] - ee_pos[t] (and ee_yaw[t+1] - ee_yaw[t] when the
    episode has yaw data).  The last step repeats the previous delta so the
    output length matches the input.  Yaw differences are wrapped to [-π, π].

    This avoids the MPC look-ahead artefact during gripper pause windows:
    because the robot is physically stationary, the observed displacement is
    naturally near-zero.
    """
    ee_pos = np.asarray(data["ee_pos"])[keep]        # (T, 3)
    dxyz   = np.diff(ee_pos, axis=0)                 # (T-1, 3)
    dxyz   = np.vstack([dxyz, dxyz[-1:]])            # (T, 3)  — pad last step

    if "ee_yaw" in data and np.asarray(data["ee_yaw"]).size > 0:
        ee_yaw = np.asarray(data["ee_yaw"])[keep]    # (T,)
        dyaw   = np.diff(ee_yaw)                     # (T-1,)
        # dyaw   = (dyaw + np.pi) % (2 * np.pi) - np.pi   # wrap to [-π, π]
        dyaw   = np.append(dyaw, dyaw[-1])           # (T,)  — pad last step
        return np.concatenate([dxyz, dyaw[:, None]], axis=-1)   # (T, 4)

    return dxyz                                       # (T, 3)


def _pause_mask(gripper_pos: np.ndarray) -> np.ndarray:
    """Boolean mask that is True for every step inside a gripper pause window."""
    settle_steps = int(round(GRASP_SETTLE_SEC * RATE_HZ))
    resume_steps = int(round(GRASP_RESUME_SEC * RATE_HZ))
    is_closed    = gripper_pos > GRIPPER_OPEN_MAX
    T            = len(gripper_pos)
    mask         = np.zeros(T, dtype=bool)
    for t in range(1, T):
        if is_closed[t] != is_closed[t - 1]:
            lo = max(0,     t - settle_steps)
            hi = min(T - 1, t + resume_steps)
            mask[lo : hi + 1] = True
    return mask


def zero_pause_actions(actions: np.ndarray, gripper_pos: np.ndarray) -> np.ndarray:
    """Replace actions inside gripper pause windows with zeros."""
    actions = actions.copy()
    actions[_pause_mask(gripper_pos)] = 0.0
    return actions


def splice_pause_with_ee(mpc_actions: np.ndarray,
                         ee_deltas: np.ndarray,
                         gripper_pos: np.ndarray) -> np.ndarray:
    """Use MPC actions everywhere; substitute EE deltas only in pause windows.

    The MPC look-ahead artefact (actions pointing upward while the robot is
    stationary) only contaminates the pause windows.  Outside those windows
    the MPC actions carry the correct scale (~10× larger than EE deltas).
    EE deltas are naturally near-zero during pauses because the robot isn't
    moving, so this splice removes the artefact without changing scale.
    """
    out  = mpc_actions.copy()
    mask = _pause_mask(gripper_pos)
    out[mask] = ee_deltas[mask]
    return out


def scale_ee_to_mpc(ee_deltas: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Multiply EE deltas by a per-dimension scale factor.

    scale should be mpc_std / ee_std (computed globally across all episodes).
    This preserves the smooth temporal shape of EE deltas — naturally
    near-zero during gripper pause windows — while bringing the magnitude
    into line with the MPC action range that the Cartesian controller expects.
    """
    return ee_deltas * scale


def relabel_pause_gripper(gripper_binarized: np.ndarray,
                          gripper_pos_raw: np.ndarray) -> np.ndarray:
    """Compute next_gripper_pos with pause-window relabeling.

    Steps inside a gripper pause window are relabeled to predict the
    POST-transition gripper state.  The window that is relabeled spans:

        [t - floor(RELABEL_SETTLE_FRACTION * settle_steps),
         t + resume_steps]

    where t is the first step of the new gripper state.

    RELABEL_SETTLE_FRACTION = 0.0  → resume half only (starts at t)
    RELABEL_SETTLE_FRACTION = 1.0  → full window (settle + resume)

    Resume-only (0.0) prevents the policy from triggering early; the full
    window (1.0) fixes a larger class imbalance at the cost of the policy
    learning to fire GRASP_SETTLE_SEC steps prematurely.
    """
    next_g       = np.pad(gripper_binarized[1:], (0, 1), mode='edge').copy()
    is_closed    = gripper_pos_raw > GRIPPER_OPEN_MAX
    settle_steps = int(round(GRASP_SETTLE_SEC * RATE_HZ))
    resume_steps = int(round(GRASP_RESUME_SEC * RATE_HZ))
    back_steps   = int(round(RELABEL_SETTLE_FRACTION * settle_steps))
    T            = len(gripper_pos_raw)

    for t in range(1, T):
        if is_closed[t] != is_closed[t - 1]:
            target = +1.0 if is_closed[t] else -1.0   # post-transition state
            lo = max(0,     t - back_steps)
            hi = min(T - 1, t + resume_steps)
            next_g[lo : hi + 1] = target

    return next_g


def filter_termination_segments(termination):
    keep = np.zeros(len(termination), dtype=bool)
    prev = 0

    for i, t in enumerate(termination):
        if t == 0:
            keep[i] = True
        elif t == 1 and prev == 0:
            keep[i] = True
        prev = t

    return keep


def apply_trim(keep: np.ndarray, data: dict) -> np.ndarray:
    """AND keep with ~pause_mask when TRIM_PAUSE_WINDOWS is True.

    The pause mask is computed on the raw (un-filtered) gripper_pos array so
    that transition points are detected correctly before any steps are removed.
    """
    if not TRIM_PAUSE_WINDOWS:
        return keep
    gripper_raw = np.asarray(data["gripper_pos"])[:keep.size]
    return keep & ~_pause_mask(gripper_raw)


def ensure_2d(x):
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    return x


def normalize(actions, amin, amax):
    denom = np.where((amax - amin) == 0, 1, (amax - amin))
    return 2 * (actions - amin) / denom - 1


def resize_obs(obs: np.ndarray, size: int) -> np.ndarray:
    """Resize (T, H, W, C) uint8 images to (T, size, size, C)."""
    out = np.empty((len(obs), size, size, obs.shape[-1]), dtype=obs.dtype)
    for i, frame in enumerate(obs):
        out[i] = np.asarray(Image.fromarray(frame).resize((size, size), Image.BILINEAR))
    return out


def build_prop(data, keep):
    """Build the proprioception array for one episode.

    Returns (prop, gripper_pos_unnorm) where:
      - prop              — the observation vector stored in the dataset;
                            gripper is normalized to [-1, +1].
      - gripper_pos_unnorm — the raw gripper_pos values (still needed for
                            action labeling, pause-mask detection, etc.).
    """
    gripper_pos_unnorm = np.asarray(data["gripper_pos"])[keep]

    # Normalize gripper to [-1, +1] for the observation.
    # Raw value is retained as gripper_pos_unnorm for all downstream logic.
    denom = max(GRIPPER_PROP_MAX - GRIPPER_PROP_MIN, 1.0)
    gripper_prop = np.clip(
        2.0 * (gripper_pos_unnorm - GRIPPER_PROP_MIN) / denom - 1.0,
        -1.0, 1.0,
    )

    try:
        parts = [
            ensure_2d(np.asarray(data["ee_pos"])[keep]),
            ensure_2d(np.asarray(data["ee_yaw"])[keep]),
            ensure_2d(gripper_prop),      # normalized, not raw
        ]
    except IndexError:
        breakpoint()

    if INCLUDE_SPAM_POSE:
        parts.append(ensure_2d(np.asarray(data["spam_pose"])[keep]))

    return np.concatenate(parts, axis=-1), gripper_pos_unnorm


# ============================================================
# PRE-PASS: compute EE→MPC scale factors (only when SCALE_EE_TO_MPC)
# ============================================================

pkl_files = sorted(input_folder.glob("*.pkl"))

ee_scale = None   # per-dimension multiplier; set below when needed

if SCALE_EE_TO_MPC:
    mpc_buf, ee_buf = [], []
    for f in pkl_files:
        with open(f, "rb") as fh:
            data = pickle.load(fh)
        term = np.asarray(data["termination"])
        keep = apply_trim(filter_termination_segments(term), data)
        mpc_buf.append(np.asarray(data["action"])[keep])
        ee_buf.append(compute_ee_delta_actions(data, keep))

    mpc_all = np.concatenate(mpc_buf, axis=0)
    ee_all  = np.concatenate(ee_buf,  axis=0)
    mpc_std = mpc_all.std(axis=0)
    ee_std  = ee_all.std(axis=0)
    # avoid division by zero for near-constant dimensions
    ee_scale = np.where(ee_std > 0, mpc_std / ee_std, 1.0)
    print(f"EE→MPC scale factors: {np.round(ee_scale, 3)}")
    print(f"  MPC std : {np.round(mpc_std, 5)}")
    print(f"  EE  std : {np.round(ee_std,  5)}")


# ============================================================
# PASS 1: collect actions for global normalization
# ============================================================

all_actions = []

for f in pkl_files:
    with open(f, "rb") as fh:
        data = pickle.load(fh)

    term = np.asarray(data["termination"])
    keep = apply_trim(filter_termination_segments(term), data)

    if SCALE_EE_TO_MPC:
        all_actions.append(scale_ee_to_mpc(compute_ee_delta_actions(data, keep), ee_scale))
    elif USE_EE_DELTA_ACTIONS:
        all_actions.append(compute_ee_delta_actions(data, keep))
    elif SPLICE_PAUSE_WITH_EE:
        raw_gripper = np.asarray(data["gripper_pos"])[keep]
        all_actions.append(splice_pause_with_ee(
            np.asarray(data["action"])[keep],
            compute_ee_delta_actions(data, keep),
            raw_gripper,
        ))
    else:
        all_actions.append(np.asarray(data["action"])[keep])

all_actions = np.concatenate(all_actions, axis=0)

a_min = all_actions.min(axis=0)
a_max = all_actions.max(axis=0)


# ============================================================
# PASS 2: build final dataset
# ============================================================

obs_list = []
prop_list = []
act_list = []
act_unnorm_list = []
term_list = []

for f in pkl_files:

    with open(f, "rb") as fh:
        data = pickle.load(fh)

    term = np.asarray(data["termination"])
    keep = apply_trim(filter_termination_segments(term), data)

    data = { k: v[:keep.size] for k, v in data.items() }

    # -------------------------
    # proprioception
    # -------------------------
    prop, gripper_pos_unnorm = build_prop(data, keep)

    # -------------------------
    # observations
    # -------------------------
    if OBS_MODE == "image":
        obs = np.asarray(data["img"])[keep]
        if RESIZE_IMAGES:
            obs = resize_obs(obs, IMAGE_SIZE)
    else:  # "proprio"
        obs = prop

    # -------------------------
    # actions
    # -------------------------
    gripper_pos_mid = 40  # threshold between open (~3) and closed (80+)
    gripper_pos = np.where(gripper_pos_unnorm < gripper_pos_mid, -1.0, 1.0)

    if SCALE_EE_TO_MPC:
        act_unnorm = scale_ee_to_mpc(compute_ee_delta_actions(data, keep), ee_scale)
    elif USE_EE_DELTA_ACTIONS:
        act_unnorm = compute_ee_delta_actions(data, keep)
    elif SPLICE_PAUSE_WITH_EE:
        act_unnorm = splice_pause_with_ee(
            np.asarray(data["action"])[keep],
            compute_ee_delta_actions(data, keep),
            gripper_pos_unnorm,   # raw, not binarised — for accurate transition detection
        )
    else:
        act_unnorm = np.asarray(data["action"])[keep]
        if ZERO_PAUSE_ACTIONS:
            act_unnorm = zero_pause_actions(act_unnorm, gripper_pos_unnorm)
    act = normalize(act_unnorm, a_min, a_max)

    if RELABEL_PAUSE_GRIPPER:
        next_gripper_pos = relabel_pause_gripper(gripper_pos, gripper_pos_unnorm)
    else:
        next_gripper_pos = np.pad(gripper_pos[1:], (0, 1), mode='edge')
    act_unnorm = np.concatenate((act_unnorm, next_gripper_pos[:, None]), axis=-1)
    act = np.concatenate((act, next_gripper_pos[:, None]), axis=-1)

    # -------------------------
    # termination
    # -------------------------
    term_f = term[keep]

    # -------------------------
    # append
    # -------------------------
    obs_list.append(obs)
    prop_list.append(prop)
    act_list.append(act)
    act_unnorm_list.append(act_unnorm)
    term_list.append(term_f)


# ============================================================
# FINAL CONCATENATION
# ============================================================

processed_dataset = {
    "observations": np.concatenate(obs_list, axis=0),
    "proprioception": np.concatenate(prop_list, axis=0),
    "actions": np.concatenate(act_list, axis=0),
    "unnormalized_actions": np.concatenate(act_unnorm_list, axis=0),
    "terminals": np.concatenate(term_list, axis=0),
    "action_min": a_min,
    "action_max": a_max,
}
if ee_scale is not None:
    processed_dataset["ee_scale"] = ee_scale


# ============================================================
# SAVE
# ============================================================

with open(output_file, "wb") as f:
    pickle.dump(processed_dataset, f)

action_mode = ("scale_ee_to_mpc" if SCALE_EE_TO_MPC
               else "ee_delta"     if USE_EE_DELTA_ACTIONS
               else "splice"       if SPLICE_PAUSE_WITH_EE
               else "zero_pause"   if ZERO_PAUSE_ACTIONS
               else "mpc")
print(f"Saved dataset  (obs_mode={OBS_MODE!r}, action_mode={action_mode!r}, "
      f"trim_pause={TRIM_PAUSE_WINDOWS}, relabel_pause={RELABEL_PAUSE_GRIPPER}, "
      f"settle_frac={RELABEL_SETTLE_FRACTION}, include_spam_pose={INCLUDE_SPAM_POSE})")
print("Shapes:")
for k, v in processed_dataset.items():
    if isinstance(v, np.ndarray):
        print(f"  {k}: {v.shape}")
