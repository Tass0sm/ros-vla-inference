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
output_file = Path("processed_dataset_state_based_v8.pkl")

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

RATE_HZ            = 10.0          # recording rate used during collection
GRASP_SETTLE_SEC   = 1.0           # _grasp_settle_sec in the demo node
GRASP_RESUME_SEC   = 1.0           # _grasp_pause_after_cmd_sec in the demo node
GRIPPER_OPEN_MAX   = 20            # gripper_pos values ≤ this are "open"

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
    """Build the proprioception array for one episode."""
    gripper_pos_unnorm = np.asarray(data["gripper_pos"])[keep]

    try:
        parts = [
            ensure_2d(np.asarray(data["ee_pos"])[keep]),
            ensure_2d(np.asarray(data["ee_yaw"])[keep]),
            ensure_2d(gripper_pos_unnorm),
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
        keep = filter_termination_segments(term)
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
    keep = filter_termination_segments(term)

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
    keep = filter_termination_segments(term)

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
      f"include_spam_pose={INCLUDE_SPAM_POSE})")
print("Shapes:")
for k, v in processed_dataset.items():
    if isinstance(v, np.ndarray):
        print(f"  {k}: {v.shape}")
