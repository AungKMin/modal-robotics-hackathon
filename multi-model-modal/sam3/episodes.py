"""
Run SAM 3 over EgoVerse zarr episodes held in a Modal Volume.

    modal run sam3/episodes.py --prompts "cup,saucer" --stride 5

Reads episodes from the `egoverse-episodes` Volume rather than shipping frames over the
wire, per the "Volumes cache zarr" convention — 816MB of episodes uploaded once, warm for
every subsequent run, and `.map()` fans out across them.

Upload the episodes first (one time):

    modal volume create egoverse-episodes
    modal volume put egoverse-episodes /home/asubuntu/egoverse_data

This file is deliberately self-contained. Importing shared constants from sam3/app.py fails
at container start: `modal run` on a file path mounts that single file to /root/, so there
is no `sam3` package to import from. Keeping it standalone means either invocation form
works. The cost is that the pins below must be kept in step with sam3/app.py by hand.
"""

import json
from typing import Optional

import modal

app = modal.App(name="sam3-egoverse-episodes")

CACHE_DIR = "/cache"
EPISODES_DIR = "/episodes"
MODEL_ID = "jetjodh/sam3"
MODEL_REVISION_ID = "1aa50ce07302cb375f85d8084b68a0fb378b8d85"
MAX_CONTAINERS = 4

cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
episodes_volume = modal.Volume.from_name("egoverse-episodes", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])

# Mirrors sam3/app.py's image, plus zarr. EgoVerse writes zarr v3 (`zarr_format: 3`) with
# sharded vlen-bytes + zstd codecs, so the v2 reader will not open these at all.
episode_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.10.0",
        "torchvision==0.25.0",
        "transformers==5.15.0",
        "kernels",
        "accelerate>=1.11.0",
        "av>=15.0.0",
        "numpy>=2.0,<3",
        "pillow>=11.0.0",
        "zarr>=3.0.0",
        "numcodecs>=0.13.0",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HUB_CACHE": CACHE_DIR})
)

with episode_image.imports():
    import io

    import numpy as np
    import torch
    import zarr
    from PIL import Image as PILImage

    from transformers import Sam3VideoModel, Sam3VideoProcessor



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

@app.function(image=episode_image, volumes={EPISODES_DIR: episodes_volume}, timeout=300)
def list_episodes() -> list[dict]:
    """
    Enumerate episodes in the Volume with their metadata.

    Runs remotely rather than listing a local mirror, so the batch reflects what is actually
    uploaded. Cheap: it only reads each episode's root zarr.json.
    """
    from pathlib import Path

    out = []
    for path in sorted(_episodes_root().iterdir()):
        meta_file = path / "zarr.json"
        if not meta_file.is_file():
            continue
        attrs = json.loads(meta_file.read_text()).get("attributes", {})
        out.append(
            {
                "episode": path.name,
                "task_name": attrs.get("task_name"),
                "task_description": attrs.get("task_description"),
                "total_frames": attrs.get("total_frames"),
                "fps": attrs.get("fps"),
                "embodiment": attrs.get("embodiment"),
            }
        )
    return out


@app.cls(
    image=episode_image,
    gpu="L40S",
    # Episodes run to 3010 frames; even subsampled this is far longer than a 30-frame clip.
    timeout=3600,
    volumes={CACHE_DIR: cache_volume, EPISODES_DIR: episodes_volume},
    secrets=[hf_secret],
    scaledown_window=300,
    max_containers=MAX_CONTAINERS,
)
class EpisodeSegmenter:
    @modal.enter()
    def load_model(self):
        self.model = Sam3VideoModel.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION_ID, device_map="cuda"
        )
        self.processor = Sam3VideoProcessor.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION_ID
        )

    def _read_frames(self, episode: str, camera: str, stride: int, max_frames: Optional[int]):
        """
        Decode a subsampled frame sequence out of the episode's zarr store.

        The image arrays are `variable_length_bytes` — one encoded image per element, not a
        raw pixel array — so each frame needs decoding before it is a picture.

        Returns (frames, source_indices, attrs). `source_indices` is the load-bearing part:
        SAM 3 indexes the sequence it is handed 0..N-1, so without this mapping a detected
        event cannot be located in the original episode timeline.
        """
        root = zarr.open_group(str(_episodes_root() / episode), mode="r")
        attrs = dict(root.attrs)
        arr = root[camera]

        # zarr v3 Arrays have no len(); use shape. total_frames is authoritative per
        # upstream (the array is padded: 999 frames, shape [1000]).
        n_stored = int(arr.shape[0])
        n = min(int(attrs.get("total_frames", n_stored)), n_stored)
        source_indices = list(range(0, n, max(1, stride)))
        if max_frames:
            source_indices = source_indices[:max_frames]
        if not source_indices:
            raise ValueError(f"{episode}: no frames to read (total_frames={n})")

        # Fancy-index once. Scalar `arr[i]` returns a doubly wrapped 0-d object ndarray, and
        # `bytes()` of that is the 8-byte pointer, not the JPEG — decode fails or, worse,
        # succeeds on garbage. List indexing returns plain `bytes` elements in a single read.
        blobs = arr[source_indices]
        frames = np.stack(
            [np.array(PILImage.open(io.BytesIO(b)).convert("RGB")) for b in blobs]
        )
        return frames, source_indices, attrs

    @modal.method()
    def segment_episode(
        self,
        episode: str,
        prompts: list[str],
        camera: str = "images.front_1",
        stride: int = 5,
        max_frames: Optional[int] = None,
        return_masks: bool = False,
    ) -> dict:
        frames, source_indices, attrs = self._read_frames(
            episode, camera, stride, max_frames
        )
        print(f"{episode}: {len(frames)} frames (stride {stride}), prompts={prompts}")

        session = self.processor.init_video_session(
            video=frames,
            inference_device="cuda",
            processing_device="cpu",
            video_storage_device="cpu",
        )
        self.processor.add_text_prompt(session, prompts)

        per_frame = {}
        for model_outputs in self.model.propagate_in_video_iterator(inference_session=session):
            out = self.processor.postprocess_outputs(session, model_outputs)
            per_frame[model_outputs.frame_idx] = _summarise(out, return_masks)

        torch.cuda.empty_cache()

        task_name = attrs.get("task_name") or ""
        return {
            "episode": episode,
            "camera": camera,
            "prompts": prompts,
            "task_name": task_name,
            "task_description": attrs.get("task_description"),
            "fps": attrs.get("fps"),
            "total_frames": attrs.get("total_frames"),
            "stride": stride,
            # Ground truth carried straight through from task_name, so eval never has to
            # re-derive it. None when the suffix is absent rather than guessing.
            "label": (
                "success" if task_name.endswith("_success")
                else "failure" if task_name.endswith("_failure")
                else None
            ),
            # sampled_frame_idx -> original episode frame number
            "source_indices": source_indices,
            "frames": per_frame,
        }


def _summarise(out: dict, return_masks: bool) -> dict:
    """Reduce one frame's masks to the scalars the geometric track consumes."""
    object_ids = out["object_ids"].tolist()
    scores = out["scores"].tolist()
    boxes = out["boxes"].cpu().numpy()
    masks = out["masks"].cpu().numpy().astype(bool)

    objects = []
    for i, obj_id in enumerate(object_ids):
        mask = masks[i]
        area = int(mask.sum())
        if area:
            ys, xs = np.nonzero(mask)
            centroid = [float(xs.mean()), float(ys.mean())]
        else:
            centroid = None
        record = {
            "object_id": int(obj_id),
            "score": float(scores[i]),
            "box_xyxy": boxes[i].tolist(),
            "area": area,
            "centroid_xy": centroid,
        }
        if return_masks:
            record["mask_packed"] = np.packbits(mask).tobytes()
            record["mask_shape"] = list(mask.shape)
        objects.append(record)

    return {
        "objects": objects,
        "prompt_to_obj_ids": {
            k: [int(v) for v in vs] for k, vs in out.get("prompt_to_obj_ids", {}).items()
        },
    }


@app.local_entrypoint()
def main(
    prompts: str = "cup,saucer",
    camera: str = "images.front_1",
    stride: int = 5,
    max_frames: int = 0,
    limit: int = 0,
    out: str = "episode_out",
    return_masks: bool = False,
):
    """
    Fan out over every episode in the Volume.

        modal run sam3/episodes.py --prompts "cup,saucer" --stride 5
        modal run sam3/episodes.py --limit 2 --max-frames 40      # quick shakeout first

    stride subsamples in time: the 20-episode slice is 21,072 frames total, so stride 5
    means ~4,200 frames of actual inference. Run --limit 2 before committing to the batch.

    prompts should be the object nouns from the episode's task_description. The default
    matches this slice, where all 20 episodes are cup_on_saucer; a broader slice needs the
    nouns derived per-episode rather than passed once.
    """
    from pathlib import Path

    episodes = list_episodes.remote()
    if not episodes:
        raise SystemExit(
            "No episodes in the Volume. Upload them first:\n"
            "  modal volume create egoverse-episodes\n"
            "  modal volume put egoverse-episodes /home/asubuntu/egoverse_data"
        )
    if limit:
        episodes = episodes[:limit]

    prompt_list = [p.strip() for p in prompts.split(",") if p.strip()]
    names = [e["episode"] for e in episodes]
    sampled = sum(len(range(0, e["total_frames"] or 0, max(1, stride))) for e in episodes)
    print(
        f"{len(episodes)} episodes, prompts={prompt_list}, stride={stride}\n"
        f"~{sampled} frames of inference across {MAX_CONTAINERS} containers"
    )

    out_dir = Path(out)
    out_dir.mkdir(exist_ok=True)

    results = EpisodeSegmenter().segment_episode.map(
        names,
        kwargs={
            "prompts": prompt_list,
            "camera": camera,
            "stride": stride,
            "max_frames": max_frames or None,
            "return_masks": return_masks,
        },
        return_exceptions=True,
    )

    ok = 0
    for name, result in zip(names, results):
        if isinstance(result, Exception):
            print(f"  {name}: FAILED — {result}")
            continue
        tracks = {o["object_id"] for f in result["frames"].values() for o in f["objects"]}
        if return_masks:
            _write_masks(out_dir, name, result)
        print(
            f"  {name}: {len(result['frames'])} frames, {len(tracks)} tracks, "
            f"label={result['label']}"
        )
        (out_dir / f"{name}.json").write_text(json.dumps(result))
        ok += 1

    print(f"✓ {ok}/{len(names)} episodes written to {out_dir}/")


def _write_masks(out_dir, stem, result) -> str:
    """Same .bin sidecar format sam3/visualize.py already reads."""
    path = out_dir / f"{stem}.masks.bin"
    offset = 0
    with open(path, "wb") as fh:
        for frame in result["frames"].values():
            for obj in frame["objects"]:
                packed = obj.pop("mask_packed", None)
                if packed is None:
                    continue
                fh.write(packed)
                obj["mask_offset"] = offset
                obj["mask_nbytes"] = len(packed)
                offset += len(packed)
    return path.name
