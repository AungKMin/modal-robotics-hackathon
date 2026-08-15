"""
Geometric track [A] for the eva_bimanual embodiment — proprioception, no vision model.

    modal run geometric/app.py
    modal run geometric/app.py --limit 2

The README's stage [A] assumes human egocentric video: MANO hand keypoints for grasp
aperture, Dyn-HaMR to factor out head motion. The eva slice has neither — it is two robot
arms on a fixed camera. But it ships something strictly better for the same purpose:
`obs_gripper` IS the aperture, measured rather than estimated, and `cmd_gripper` gives the
commanded target alongside it.

That pairing is the key adaptation, and it needs no vision at all. A parallel gripper
commanded fully closed stops early when there is something between the fingers, so
`obs - cmd` plateaus at the object's width while it is held. From that:

    holding  = commanded closed AND obs settled above the closed-on-air band
    release  = holding ends because cmd opened            (intentional)
    slip     = holding ends while cmd stays closed        (the object left on its own)

Slip is the robot-side drop signature. It replaces "aperture opens while object velocity
does not match" and is far less ambiguous.

Conventions verified against the data, not assumed (see README):
  - gripper: 0 = closed, 1 = open   (checked on the wrist camera images)
  - each arm's obs_ee_pose is in its OWN base frame; `extrinsics[arm]` is the camera pose
    in that frame, so p_cam = inv(extrinsics[arm]) @ p_arm puts both arms in one frame
    (verified by projecting through the intrinsics onto the front camera image)

Runs on CPU. The whole 20-episode slice costs cents and seconds, which makes it the right
backbone for a prevalence sweep — the expensive VLM only has to adjudicate what this flags.
"""

import json
from typing import Optional

import modal

app = modal.App(name="geometric-track")

EPISODES_DIR = "/episodes"
# Which Volume holds the episodes. Override per run, e.g. EPISODES_VOLUME=egoverse-cup50
# (read at import time; `modal run` imports this file locally, so a shell env var works).
import os as _os
EPISODES_VOLUME = _os.environ.get("EPISODES_VOLUME", "egoverse-episodes")
episodes_volume = modal.Volume.from_name(EPISODES_VOLUME, create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "numpy>=2.0,<3", "zarr>=3.0.0", "numcodecs>=0.13.0"
)

with image.imports():
    import numpy as np
    import zarr

ARMS = ("left", "right")
# Gripper values are normalised to [0, 1] with 0 = closed. Midpoint splits open/closed.
OPEN_THRESH = 0.5
# Closed-on-nothing reads 0.000-0.006 (fingers touching on the wrist cam). Blocked on the
# cup body plateaus at 0.026-0.101; pinching the cup RIM reads as low as 0.017. So a settled
# reading above HOLD_EPS while commanded closed means something is between the fingers, a
# reading at or below AIR_EPS means nothing is, and (AIR_EPS, HOLD_EPS] is ambiguous —
# reported as such rather than forced either way. Fixed thresholds are deliberate: a
# per-episode floor fails whenever an arm is only ever closed while holding, which happens.
HOLD_EPS = 0.01
AIR_EPS = 0.006
# A hold is a settled plateau, not a transit. Closing on nothing sweeps obs from ~0.25 to
# 0 in ~6 frames and would otherwise register as a 6-frame "hold" ending in a "slip".
SETTLED_DELTA = 0.01     # |obs[t] - obs[t-3]| below this = fingers have stopped moving
MIN_HOLD_S = 0.3         # and the plateau must last at least this long
RELEASE_WINDOW_S = 0.5   # cmd opening within this window after a hold ends = release
# Two grippers both blocked on the object within this distance = a handover is happening.
HANDOVER_DIST_M = 0.30
CAMERA = "front_1"



def _episodes_root():
    """
    Directory that directly contains the episode folders.

    `modal volume put vol /local/egoverse_data /` uploads the directory *by name*, so the
    episodes land at /episodes/egoverse_data/<ep>, not /episodes/<ep>. Accept either layout
    rather than forcing a re-upload.
    """
    from pathlib import Path

    root = Path(EPISODES_DIR)
    if not root.is_dir():
        return root
    if any((p / "zarr.json").is_file() for p in root.iterdir() if p.is_dir()):
        return root
    for sub in sorted(root.iterdir()):
        if sub.is_dir() and any((p / "zarr.json").is_file() for p in sub.iterdir() if p.is_dir()):
            return sub
    return root

@app.function(image=image, volumes={EPISODES_DIR: episodes_volume}, timeout=300)
def list_episodes() -> list[str]:
    from pathlib import Path

    return sorted(
        p.name for p in _episodes_root().iterdir() if (p / "zarr.json").is_file()
    )


def _runs(mask: "np.ndarray") -> list[tuple]:
    """Contiguous True runs in a boolean array as (start, end_exclusive) pairs."""
    if not mask.any():
        return []
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def _to_camera_frame(p_arm: "np.ndarray", extrinsic: list) -> "np.ndarray":
    """Arm-base-frame xyz -> camera-frame xyz. extrinsic is T_arm<-cam (4x4), so invert."""
    T = np.linalg.inv(np.asarray(extrinsic, dtype=np.float64))
    homo = np.c_[p_arm, np.ones(len(p_arm))]
    return (T @ homo.T).T[:, :3]


@app.function(
    image=image,
    volumes={EPISODES_DIR: episodes_volume},
    timeout=600,
    # No GPU. Proprioception is arithmetic over a few thousand floats per episode.
)
def extract_features(episode: str) -> dict:
    """
    Derive the stage [A] feature set for one episode from proprioception alone.

    Every array is read as float and treated as NaN-contaminated, because it is: three
    episodes in this slice carry all-NaN `left.cmd_gripper`, and one has NaN in
    `left.obs_gripper` too. All reductions are nan-aware, missing fields degrade to
    obs-only logic where possible, and coverage is reported in `nan_fields`.
    """
    root = zarr.open_group(str(_episodes_root() / episode), mode="r")
    attrs = dict(root.attrs)
    n = int(attrs.get("total_frames", 0))
    fps = float(attrs.get("fps") or 30.0)
    extrinsics = attrs.get("extrinsics") or {}

    def arr(name):
        return np.asarray(root[name][:n], dtype=np.float64).reshape(n, -1)

    feats: dict = {}
    nan_fields: list = []
    holding_masks: dict = {}
    cam_pos: dict = {}

    for arm in ARMS:
        obs_g = arr(f"{arm}.obs_gripper").ravel()
        cmd_g = arr(f"{arm}.cmd_gripper").ravel()
        obs_p = arr(f"{arm}.obs_ee_pose")[:, :3]  # xyz metres, arm base frame; [3:] quat
        cmd_p = arr(f"{arm}.cmd_ee_pose")[:, :3]

        for label, v in (("obs_gripper", obs_g), ("cmd_gripper", cmd_g),
                         ("obs_ee_pose", obs_p), ("cmd_ee_pose", cmd_p)):
            if np.isnan(v).any():
                nan_fields.append(f"{arm}.{label}")

        obs_ok = ~np.isnan(obs_g)
        cmd_ok = ~np.isnan(cmd_g)
        cmd_usable = cmd_ok.any()

        # --- gripper state ------------------------------------------------------------
        closed = obs_ok & (obs_g < OPEN_THRESH)
        cmd_closed = (cmd_ok & (cmd_g < OPEN_THRESH)) if cmd_usable else closed

        # Diagnostic only: the lowest settled reading while commanded closed. NOT used as a
        # baseline for hold detection (see HOLD_EPS) — kept so a drifting gripper is visible.
        floor_pool = obs_g[cmd_closed & obs_ok] if (cmd_closed & obs_ok).any() else obs_g[obs_ok]
        floor = float(np.percentile(floor_pool, 5)) if floor_pool.size else float("nan")

        # Blocked by an object: commanded closed, fingers stopped above the air band, and
        # stopped moving. The settled test is what separates a plateau from a transit.
        d3 = np.abs(obs_g - np.roll(obs_g, 3))
        d3[:3] = np.inf
        settled = d3 < SETTLED_DELTA
        holding_raw = cmd_closed & obs_ok & (obs_g > HOLD_EPS) & closed & settled
        min_hold = max(1, int(MIN_HOLD_S * fps))
        hold_runs = [(s, e) for s, e in _runs(holding_raw) if e - s >= min_hold]
        holding = np.zeros(n, dtype=bool)
        for s, e in hold_runs:
            holding[s:e] = True
        holding_masks[arm] = holding

        # How each hold ended. cmd opening shortly after = release (intentional). cmd still
        # closed AND obs down in the air band = slip (the object left on its own). cmd still
        # closed but obs only in the ambiguous band = unresolved; counted, not called a slip.
        slips, releases, ambiguous = 0, 0, 0
        slip_frames = []
        win = max(1, int(RELEASE_WINDOW_S * fps))
        for s, e in hold_runs:
            if e >= n:
                continue  # still holding at episode end
            cmd_after = cmd_g[e:min(n, e + win)]
            obs_after = obs_g[e:min(n, e + win)]
            if cmd_usable and (cmd_after >= OPEN_THRESH).any():
                releases += 1
            elif (obs_after <= AIR_EPS).any():
                slips += 1
                slip_frames.append(int(e))
            else:
                ambiguous += 1

        # --- kinematics ---------------------------------------------------------------
        # Speed in m/s. gradient keeps the array length so it stays index-aligned.
        speed = np.linalg.norm(np.gradient(obs_p, axis=0), axis=1) * fps
        # Tracking error: sustained gap means the controller is fighting something.
        track_err = np.linalg.norm(obs_p - cmd_p, axis=1)

        if arm in extrinsics:
            cam_pos[arm] = _to_camera_frame(obs_p, extrinsics[arm])

        feats[arm] = {
            "closed_reading_p5": None if np.isnan(floor) else floor,
            "closed_fraction": _nanmean(closed.astype(float)),
            "hold_fraction": _nanmean(holding.astype(float)),
            "n_holds": len(hold_runs),
            "longest_hold_s": max((e - s for s, e in hold_runs), default=0) / fps,
            "hold_width_median": (
                float(np.median(obs_g[holding])) if holding.any() else None
            ),
            "n_slips": slips,
            "n_releases": releases,
            "n_ambiguous_ends": ambiguous,
            "first_slip_frame": slip_frames[0] if slip_frames else None,
            "cmd_gripper_available": bool(cmd_usable),
            "ee_speed_mean": _nanmean(speed),
            "ee_speed_max": _nanmax(speed),
            "ee_path_length_m": float(
                np.nansum(np.linalg.norm(np.diff(obs_p, axis=0), axis=1))
            ),
            "track_err_mean": _nanmean(track_err),
            "track_err_max": _nanmax(track_err),
        }

    # --- bimanual -----------------------------------------------------------------------
    # "pick up cup with one arm, hand it over to the other arm, place it on the saucer":
    # a handover is both grippers blocked on the object at the same time, close together.
    both_holding = holding_masks["left"] & holding_masks["right"]
    if "left" in cam_pos and "right" in cam_pos:
        hand_distance = np.linalg.norm(cam_pos["left"] - cam_pos["right"], axis=1)
    else:
        hand_distance = np.full(n, np.nan)
    handover_mask = both_holding & (hand_distance < HANDOVER_DIST_M)
    handover_runs = _runs(handover_mask)
    longest = max(((e - s) for s, e in handover_runs), default=0)
    handover_frame = None
    if handover_runs:
        s, e = max(handover_runs, key=lambda se: se[1] - se[0])
        handover_frame = int((s + e) // 2)

    any_hold = holding_masks["left"] | holding_masks["right"]
    feats["bimanual"] = {
        "both_holding_fraction": _nanmean(both_holding.astype(float)),
        "longest_handover_s": longest / fps,
        "n_handover_candidates": len(handover_runs),
        "handover_frame": handover_frame,
        "handover_detected": bool(longest > 0),
        "min_hand_distance_m": _nanmin(hand_distance),
        "hand_distance_at_handover_m": (
            float(hand_distance[handover_frame]) if handover_frame is not None else None
        ),
        # Object held by either arm at any point? An episode where nothing was ever
        # grasped cannot have succeeded, whatever the video shows.
        "any_hold": bool(any_hold.any()),
        "total_slips": feats["left"]["n_slips"] + feats["right"]["n_slips"],
        # Frame of the last hold ending anywhere — a proxy for "when the object was set
        # down", useful for aligning the VLM trace to the physical end of the attempt.
        "last_hold_end_frame": (
            int(max(e for arm in ARMS for s, e in _runs(holding_masks[arm]))) if any_hold.any() else None
        ),
    }

    task_name = attrs.get("task_name") or ""
    return {
        "episode": episode,
        "task_name": task_name,
        "task_description": attrs.get("task_description"),
        "label": (
            "success" if task_name.endswith("_success")
            else "failure" if task_name.endswith("_failure")
            else None
        ),
        "total_frames": n,
        "fps": fps,
        "nan_fields": sorted(set(nan_fields)),
        "features": feats,
    }


def _nanmean(v) -> Optional[float]:
    v = np.asarray(v, dtype=np.float64)
    return None if np.isnan(v).all() else float(np.nanmean(v))


def _nanmax(v) -> Optional[float]:
    v = np.asarray(v, dtype=np.float64)
    return None if np.isnan(v).all() else float(np.nanmax(v))


def _nanmin(v) -> Optional[float]:
    v = np.asarray(v, dtype=np.float64)
    return None if np.isnan(v).all() else float(np.nanmin(v))


@app.local_entrypoint()
def main(limit: int = 0, out: str = "geometric_out"):
    """
    Extract proprioception features for every episode in the Volume.

        modal run geometric/app.py

    CPU-only and fans out per episode, so the whole slice finishes in seconds. Prints a
    per-feature separation check against the labels in task_name.
    """
    from pathlib import Path

    episodes = list_episodes.remote()
    if not episodes:
        raise SystemExit(
            "No episodes in the Volume. Upload them first:\n"
            "  modal volume put egoverse-episodes /home/asubuntu/egoverse_data"
        )
    if limit:
        episodes = episodes[:limit]

    out_dir = Path(out)
    out_dir.mkdir(exist_ok=True)

    results = list(extract_features.map(episodes, return_exceptions=True))

    rows = []
    for name, res in zip(episodes, results):
        if isinstance(res, Exception):
            print(f"  {name}: FAILED — {res}")
            continue
        (out_dir / f"{name}.json").write_text(json.dumps(res, indent=1))
        rows.append(res)
        print("  " + _summary_line(res))

    _separation(rows)
    print(f"\n✓ {len(rows)}/{len(episodes)} feature sets written to {out_dir}/")


def _summary_line(res: dict) -> str:
    bi = res["features"]["bimanual"]
    l, r = res["features"]["left"], res["features"]["right"]
    warn = f"  ⚠ NaN: {','.join(res['nan_fields'])}" if res["nan_fields"] else ""
    return (
        f"{res['episode']}  {str(res['label']):<8} "
        f"holds L{l['n_holds']}/R{r['n_holds']}  slips={bi['total_slips']}  "
        f"handover={str(bi['handover_detected']):<5} ({bi['longest_handover_s']:.2f}s)  "
        f"minDist={bi['min_hand_distance_m'] if bi['min_hand_distance_m'] is None else round(bi['min_hand_distance_m'], 3)}m{warn}"
    )


def _separation(rows: list[dict]) -> None:
    """
    Rank features by how well they separate the labels already in the data.

    This is the point of having 10/10 labelled episodes: it says which features are worth
    feeding the fusion stage before any model is trained on them. Reported as a normalised
    mean gap, not a p-value — n=20 does not support significance claims.
    """
    import statistics as st

    labelled = [r for r in rows if r["label"]]
    if len(labelled) < 4:
        return

    flat = []
    for r in labelled:
        row = {}
        for group, feats in r["features"].items():
            for k, v in feats.items():
                if isinstance(v, bool):
                    row[f"{group}.{k}"] = float(v)
                elif isinstance(v, (int, float)):
                    row[f"{group}.{k}"] = float(v)
        flat.append((r["label"], row))

    keys = set(flat[0][1])
    for _, row in flat:
        keys &= set(row)

    scored = []
    for k in sorted(keys):
        succ = [row[k] for lab, row in flat if lab == "success"]
        fail = [row[k] for lab, row in flat if lab == "failure"]
        if not succ or not fail:
            continue
        pooled = st.pstdev(succ + fail)
        if pooled < 1e-12:
            continue
        gap = (st.mean(succ) - st.mean(fail)) / pooled
        scored.append((abs(gap), gap, k, st.mean(succ), st.mean(fail)))

    scored.sort(reverse=True)
    print(f"\nFeature separation (success - failure, in pooled SDs), n={len(labelled)}:")
    for _, gap, k, ms, mf in scored[:12]:
        print(f"  {k:<36} {gap:+.2f}   success {ms:>9.3f}   failure {mf:>9.3f}")
