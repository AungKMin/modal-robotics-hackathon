"""
Pull somundane/egoverse-cup50 from the Hub and lay it out as per-episode zarr groups in a
Modal Volume, in the same shape the SAM 3 / VLM apps already read from the eva slice.

    modal run cup50/prepare.py

The dataset is one parquet: (episode, frame, image, left/right_obs_ee_pose, obs_head_pose,
obs_eye_gaze) at 1 fps (frame = 0, 30, 60, ...). It is egocentric human data — head pose
and eye gaze, no gripper — so the geometric track does not apply, and it carries no
success/failure labels. That makes it the prevalence-audit set: tag every episode, report
the failure fraction.

After this, point any app at the new Volume with an env var (read at import time):

    EPISODES_VOLUME=egoverse-cup50 modal run vlm_critic/app.py --stride 1 --out vlm_critic_out_cup50
    EPISODES_VOLUME=egoverse-cup50 modal run sam3/episodes.py --stride 1 --prompts "cup,saucer" --out sam3_out/cup50

Frames are already 1 fps, so --stride 1 uses every frame that exists.
"""

import json

import modal

app = modal.App(name="cup50-prepare")

REPO_ID = "somundane/egoverse-cup50"
PARQUET = "data/train-00000-of-00001.parquet"
VOLUME_NAME = "egoverse-cup50"
OUT_DIR = "/episodes"
TASK_DESCRIPTION = "pick up the cup and place it on the saucer"

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])

image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "huggingface-hub>=0.30", "pyarrow>=17", "numpy>=2.0,<3", "zarr>=3.0.0", "numcodecs>=0.13.0"
)

with image.imports():
    import numpy as np
    import pyarrow.parquet as pq
    import zarr
    from huggingface_hub import hf_hub_download


@app.function(image=image, volumes={OUT_DIR: volume}, secrets=[hf_secret], timeout=1800)
def prepare(force: bool = False) -> dict:
    from pathlib import Path

    path = hf_hub_download(REPO_ID, PARQUET, repo_type="dataset")
    table = pq.read_table(path)
    cols = table.column_names
    print("columns:", cols, "| rows:", table.num_rows)

    episodes = table.column("episode").to_pylist()
    frames = table.column("frame").to_pylist()
    images = table.column("image").to_pylist()  # dicts with 'bytes' (and 'path')
    pose_cols = [c for c in cols if c.endswith("_pose") or c.endswith("_gaze")]
    poses = {c: table.column(c).to_pylist() for c in pose_cols}

    by_ep: dict = {}
    for i, ep in enumerate(episodes):
        by_ep.setdefault(ep, []).append(i)

    summary = {"episodes": 0, "frames": 0, "skipped": 0}
    for ep, rows in sorted(by_ep.items()):
        rows.sort(key=lambda i: frames[i])
        out = Path(OUT_DIR) / ep
        # Only skip episodes that finished; a crash mid-episode leaves a partial group.
        if not force and (out / "zarr.json").exists():
            try:
                done = zarr.open_group(str(out), mode="r").attrs.get("complete", False)
            except Exception:  # noqa: BLE001
                done = False
            if done:
                summary["skipped"] += 1
                continue
        blobs = []
        for i in rows:
            img = images[i]
            b = img.get("bytes") if isinstance(img, dict) else None
            if b is None:
                raise RuntimeError(f"{ep} row {i}: image has no bytes (keys={list(img) if isinstance(img, dict) else type(img)})")
            blobs.append(bytes(b))
        n = len(blobs)
        src_frames = [int(frames[i]) for i in rows]
        # fps of the *sampled* sequence: frames are 30 apart at 30 fps -> 1 fps.
        step = (src_frames[1] - src_frames[0]) if n > 1 else 30
        fps = 30.0 / max(1, step)

        g = zarr.open_group(str(out), mode="w")
        g.attrs.update({
            "embodiment": "aria",  # head pose + eye gaze: egocentric human
            "total_frames": n,
            "fps": fps,
            "source_frames": src_frames,   # original 30 fps frame numbers
            "task_name": "cup_on_saucer",  # no label suffix: unlabelled by design
            "task_description": TASK_DESCRIPTION,
            "source": REPO_ID,
        })
        a = g.create_array("images.front_1", shape=(n,), dtype=zarr.dtype.VariableLengthBytes(), chunks=(n,))
        a[:] = np.array(blobs, dtype=object)
        for c in pose_cols:
            vals = [poses[c][i] or [] for i in rows]
            width = max((len(v) for v in vals), default=0)
            if width == 0:
                continue  # column empty for this episode (some rows carry no pose)
            arr = np.full((n, width), np.nan, dtype=np.float32)
            for k, v in enumerate(vals):
                arr[k, : len(v)] = v
            # eva naming: left.obs_ee_pose etc. Keep obs_head_pose / obs_eye_gaze as-is.
            name = c.replace("left_obs_", "left.obs_").replace("right_obs_", "right.obs_")
            g.create_array(name, shape=arr.shape, dtype="float32", chunks=arr.shape)[:] = arr
        g.attrs["complete"] = True
        summary["episodes"] += 1
        summary["frames"] += n
        print(f"  {ep}: {n} frames @ {fps:.1f} fps")

    volume.commit()
    return summary


@app.local_entrypoint()
def main(force: bool = False):
    s = prepare.remote(force=force)
    print(json.dumps(s, indent=1))
    print(f"\n✓ Volume '{VOLUME_NAME}' ready. Next:\n"
          f"  EPISODES_VOLUME={VOLUME_NAME} modal run vlm_critic/app.py --stride 1 --out vlm_critic_out_cup50\n"
          f"  EPISODES_VOLUME={VOLUME_NAME} modal run sam3/episodes.py --stride 1 --prompts \"cup,saucer\" --out sam3_out/cup50\n"
          f"  python3 fuse/fuse.py --vlm-dir vlm_critic_out_cup50/qwen --seg-dir sam3_out/cup50 --geo-dir none")
