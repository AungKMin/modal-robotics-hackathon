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
# Rendered overlays and traces are written here as well as returned to the caller, so the
# artefacts survive in Modal storage independent of whoever ran the job.
OUTPUTS_DIR = "/outputs"
outputs_volume = modal.Volume.from_name("egoverse-outputs", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])

# Overlay palette, keyed by object id so a track keeps its colour for its whole life.
PALETTE = [
    (255, 89, 94), (56, 176, 0), (25, 130, 196), (255, 202, 58),
    (154, 78, 174), (255, 146, 76), (0, 187, 249), (241, 91, 181),
]
ALPHA = 0.45

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
    volumes={
        CACHE_DIR: cache_volume,
        EPISODES_DIR: episodes_volume,
        OUTPUTS_DIR: outputs_volume,
    },
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

    def _held_overlay_frames(self, episode, camera, n_src, source_indices, dets):
        """Yield every source frame 0..n_src-1 overlaid with the latest sampled detections."""
        root = zarr.open_group(str(_episodes_root() / episode), mode="r")
        arr = root[camera]
        sampled = sorted(dets)
        j = 0
        chunk = 64
        for start in range(0, n_src, chunk):
            idxs = list(range(start, min(n_src, start + chunk)))
            for i, blob in zip(idxs, arr[idxs]):
                while j + 1 < len(sampled) and source_indices[sampled[j + 1]] <= i:
                    j += 1
                frame = np.array(PILImage.open(io.BytesIO(blob)).convert("RGB"))
                yield _draw_overlay(frame, _unpack_dets(dets[sampled[j]]), i)

    @modal.method()
    def segment_episode(
        self,
        episode: str,
        prompts: list[str],
        camera: str = "images.front_1",
        stride: int = 5,
        max_frames: Optional[int] = None,
        return_masks: bool = False,
        render: bool = True,
        render_full_fps: bool = True,
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
        dets = {}  # sampled idx -> bit-packed masks + boxes, kept for rendering
        for model_outputs in self.model.propagate_in_video_iterator(inference_session=session):
            out = self.processor.postprocess_outputs(session, model_outputs)
            idx = model_outputs.frame_idx
            per_frame[idx] = _summarise(out, return_masks)
            if render:
                dets[idx] = _pack_dets(out)

        torch.cuda.empty_cache()

        fps = float(attrs.get("fps") or 30.0)
        overlay_mp4 = None
        if render and dets:
            if render_full_fps and stride > 1:
                # Every source frame at native fps; each frame wears the masks of the most
                # recent sampled frame. Smooth video, inference cost unchanged.
                n_src = source_indices[-1] + 1
                overlay_mp4 = _encode_mp4_stream(
                    self._held_overlay_frames(episode, camera, n_src, source_indices, dets),
                    int(round(fps)),
                )
            else:
                overlay_mp4 = _encode_mp4(
                    [_draw_overlay(frames[i], _unpack_dets(dets[i]), source_indices[i])
                     for i in sorted(dets)],
                    max(1, round(fps / max(1, stride))),
                )
            out_path = f"{OUTPUTS_DIR}/sam3/{episode}_overlay.mp4"
            import os
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "wb") as fh:
                fh.write(overlay_mp4)
            outputs_volume.commit()
            print(f"{episode}: overlay {len(overlay_mp4)/1e6:.1f} MB -> volume:{out_path}")

        task_name = attrs.get("task_name") or ""
        return {
            "overlay_mp4": overlay_mp4,
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


def _pack_dets(out: dict) -> list:
    """Compact per-frame detections: (id, score, box, packed mask, shape). ~38KB per mask."""
    ids = out["object_ids"].tolist()
    masks = out["masks"].cpu().numpy().astype(bool)
    boxes = out["boxes"].cpu().numpy()
    scores = out["scores"].tolist()
    return [
        (int(ids[i]), float(scores[i]), [float(v) for v in boxes[i]],
         np.packbits(masks[i]).tobytes(), masks[i].shape)
        for i in range(len(ids))
    ]


def _unpack_dets(packed: list) -> list:
    out = []
    for obj_id, score, box, pb, shape in packed:
        m = np.unpackbits(np.frombuffer(pb, dtype=np.uint8))[: shape[0] * shape[1]]
        out.append((obj_id, score, box, m.reshape(shape).astype(bool)))
    return out


def _encode_mp4_stream(frame_iter, fps: int) -> bytes:
    """Like _encode_mp4 but consumes a generator, so long episodes never sit in RAM."""
    import av

    buf = io.BytesIO()
    container = av.open(buf, mode="w", format="mp4")
    stream = None
    w = h = w2 = h2 = 0
    for f in frame_iter:
        if stream is None:
            h, w = f.shape[:2]
            w2, h2 = w + (w % 2), h + (h % 2)
            try:
                stream = container.add_stream("libx264", rate=fps)
                stream.options = {"crf": "23", "preset": "veryfast"}
            except Exception:
                stream = container.add_stream("mpeg4", rate=fps)
            stream.width, stream.height, stream.pix_fmt = w2, h2, "yuv420p"
        if (w2, h2) != (w, h):
            padded = np.zeros((h2, w2, 3), dtype=np.uint8)
            padded[:h, :w] = f
            f = padded
        vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="rgb24")
        for packet in stream.encode(vf):
            container.mux(packet)
    if stream is not None:
        for packet in stream.encode():
            container.mux(packet)
    container.close()
    return buf.getvalue()


def _draw_overlay(frame: "np.ndarray", dets: list, source_idx: int) -> "np.ndarray":
    """
    Tint each mask in its track's colour, box it, label `id:score`, stamp the source frame.
    `dets` is a list of (obj_id, score, box_xyxy, mask_bool) — see _pack_dets/_unpack_dets.
    """
    from PIL import ImageDraw

    arr = frame.astype(np.float32)
    object_ids = [d[0] for d in dets]
    scores = [d[1] for d in dets]
    boxes = [d[2] for d in dets]
    masks = [d[3] for d in dets]

    for i, obj_id in enumerate(object_ids):
        m = masks[i]
        if m.shape != arr.shape[:2]:
            continue
        tint = np.array(PALETTE[int(obj_id) % len(PALETTE)], dtype=np.float32)
        arr[m] = arr[m] * (1 - ALPHA) + tint * ALPHA

    img = PILImage.fromarray(arr.astype(np.uint8))
    draw = ImageDraw.Draw(img)
    for i, obj_id in enumerate(object_ids):
        color = PALETTE[int(obj_id) % len(PALETTE)]
        x0, y0, x1, y1 = [float(v) for v in boxes[i]]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        label = f"{int(obj_id)}:{scores[i]:.2f}"
        tx, ty = x0 + 2, max(0, y0 - 12)
        draw.rectangle([tx - 1, ty - 1, tx + 6 * len(label) + 2, ty + 11], fill=(0, 0, 0))
        draw.text((tx, ty), label, fill=color)
    draw.text((6, 6), f"frame {source_idx}", fill=(0, 0, 0))
    draw.text((5, 5), f"frame {source_idx}", fill=(255, 255, 255))
    return np.asarray(img)


def _encode_mp4(frames: list, fps: int) -> bytes:
    """
    Encode RGB frames to an in-memory H.264 mp4 with PyAV (already in the image for the
    clip path). Falls back to mpeg4 if this PyAV build lacks libx264. Odd dimensions are
    padded because yuv420p needs even width and height.
    """
    import av

    h, w = frames[0].shape[:2]
    w2, h2 = w + (w % 2), h + (h % 2)
    buf = io.BytesIO()
    with av.open(buf, mode="w", format="mp4") as container:
        try:
            stream = container.add_stream("libx264", rate=fps)
            stream.options = {"crf": "23", "preset": "veryfast"}
        except Exception:
            stream = container.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = w2, h2, "yuv420p"
        for f in frames:
            if (w2, h2) != (w, h):
                padded = np.zeros((h2, w2, 3), dtype=np.uint8)
                padded[:h, :w] = f
                f = padded
            vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="rgb24")
            for packet in stream.encode(vf):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buf.getvalue()


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
    out: str = "sam3_out/episodes",
    return_masks: bool = False,
    render: bool = True,
    render_full_fps: bool = True,
):
    """
    Fan out over every episode in the Volume.

    Outputs, per episode:
      <out>/<episode>.json          per-frame trace (objects, boxes, centroids, ids)
      <out>/<episode>_overlay.mp4   masks tinted per track, boxes, id:score, source frame
    The same mp4 is also written to the `egoverse-outputs` Volume under /sam3/, so it lives
    in Modal storage independent of the machine that ran the job. --no-render skips it.

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
    out_dir.mkdir(parents=True, exist_ok=True)

    results = EpisodeSegmenter().segment_episode.map(
        names,
        kwargs={
            "prompts": prompt_list,
            "camera": camera,
            "stride": stride,
            "max_frames": max_frames or None,
            "return_masks": return_masks,
            "render": render,
            "render_full_fps": render_full_fps,
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
        # Bytes are not JSON; pull the video out of the record before dumping the trace.
        mp4 = result.pop("overlay_mp4", None)
        video_note = ""
        if mp4:
            (out_dir / f"{name}_overlay.mp4").write_bytes(mp4)
            video_note = f", overlay {len(mp4)/1e6:.1f} MB"
        print(
            f"  {name}: {len(result['frames'])} frames, {len(tracks)} tracks, "
            f"label={result['label']}{video_note}"
        )
        (out_dir / f"{name}.json").write_text(json.dumps(result))
        ok += 1

    print(f"✓ {ok}/{len(names)} episodes written to {out_dir}/")
    if render and ok:
        print("  overlays also in Modal storage: modal volume ls egoverse-outputs /sam3")


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
