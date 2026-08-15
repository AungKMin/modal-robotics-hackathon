"""
Promptable Concept Segmentation over an episode with SAM 3.

Text noun phrase in ("the towel"), per-frame instance masks + stable object IDs out.
This is the segmentation stage of the geometric track: it upgrades hand-object distance
from keypoint proximity to mask-level contact, and makes "the object left the frame /
hit the floor" directly measurable.

The whole point of SAM 3 over SAM 2 here is that the prompt is *text*, so the object noun
parsed out of an episode's language annotation drives it with no per-episode human seeding.
"""

import io
from typing import Optional

import modal

app = modal.App(name="sam3-concept-segmentation")

CACHE_DIR = "/cache"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        # The load-bearing trio is pinned. sam3_video landed in transformers 5.x — 4.x cannot
        # import Sam3VideoModel at all, so the `transformers~=4.44` pin copied from the
        # text_to_image template hard-fails here.
        # torch 2.10 rather than 2.9 specifically so `kernels` below finds a matching prebuilt
        # variant — its wheels cover torch 2.10-2.13 only, and on 2.9 it silently no-ops.
        "torch==2.10.0",
        "torchvision==0.25.0",
        "transformers==5.15.0",
        # Without `kernels`, transformers silently skips NMS post-processing, hole filling and
        # sprinkle removal — which degrades exactly the mask area/centroid numbers the
        # geometric features are built on. Not optional here.
        "kernels",
        # Left to the resolver so it can pick versions ABI-compatible with the torch build
        # above. Freeze these once the image builds clean.
        "accelerate>=1.11.0",
        "av>=15.0.0",
        "numpy>=2.0,<3",
        "pillow>=11.0.0",
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HUB_CACHE": CACHE_DIR,
        }
    )
)

with image.imports():
    import numpy as np
    import torch

    from transformers import Sam3VideoModel, Sam3VideoProcessor

# facebook/sam3 is GATED — deploying against it needs a Modal Secret carrying HF_TOKEN
# from an account that has accepted Meta's terms. This ungated mirror is a straight
# snapshot of the same Sam3VideoModel weights (config.json architectures == Sam3VideoModel),
# same trick the text_to_image template uses for SD 3.5.
MODEL_ID = "jetjodh/sam3"
MODEL_REVISION_ID = "1aa50ce07302cb375f85d8084b68a0fb378b8d85"

# To switch to the official weights instead:
#   1. accept terms at https://huggingface.co/facebook/sam3
#   2. MODEL_ID = "facebook/sam3"; MODEL_REVISION_ID = None
# The HF_TOKEN secret attached below is what authorises the gated download.

# Authenticates Hub downloads: raises the anonymous rate limit, speeds up transfers, and is
# what unlocks gated repos like facebook/sam3. Create it once with:
#   modal secret create huggingface-secret HF_TOKEN=hf_...
hf_secret = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])

cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

# Every container that starts pays the model load in @modal.enter() once. Bounding the
# autoscaler trades wall-clock for cold starts: at 1 you get exactly one load and the clips
# run sequentially through a single warm container; at 10 all ten clips start at once but
# you pay ten loads. 4 is a reasonable middle for a batch of ~10 short clips.
MAX_CONTAINERS = 4


@app.cls(
    image=image,
    # config.json declares dtype float32, so this loads as 860M params in fp32 (~3.4GB),
    # plus 1008px activations and the tracker's memory bank. L40S has the headroom to not
    # think about it, and is what Modal's own docs recommend as the default inference GPU.
    # Drop to A10 or L4 for the fan-out sweep once you know your per-episode frame budget —
    # that is the config the prevalence audit's cost-per-1k-episodes number depends on.
    gpu="L40S",
    timeout=1800,
    volumes={CACHE_DIR: cache_volume},
    secrets=[hf_secret],
    scaledown_window=300,
    max_containers=MAX_CONTAINERS,
)
class ConceptSegmenter:
    @modal.enter()
    def load_model(self):
        """Load SAM 3 into GPU memory on container startup."""
        self.model = Sam3VideoModel.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION_ID,
            device_map="cuda",
        )
        self.processor = Sam3VideoProcessor.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION_ID,
        )

    def _decode(self, video_bytes: bytes) -> "np.ndarray":
        """Decode mp4 bytes to a (T, H, W, 3) uint8 array."""
        import av

        with av.open(io.BytesIO(video_bytes)) as container:
            return np.stack(
                [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
            )

    @modal.method()
    def segment(
        self,
        prompts: list[str],
        video_bytes: Optional[bytes] = None,
        frames: Optional["np.ndarray"] = None,
        max_frames: Optional[int] = None,
        return_masks: bool = False,
    ) -> dict:
        """
        Track every instance matching each text prompt across the episode.

        Exactly one of `frames` (T, H, W, 3) uint8 — what you get out of a zarr episode —
        or `video_bytes` (mp4) must be given.

        Returns per-frame, per-object records. Masks are summarised to area / centroid /
        bbox by default rather than shipped whole: a 500-frame episode at 1008px is ~500MB
        of bool mask per object, and the geometric features only need the summary. Pass
        return_masks=True to also get bit-packed full-resolution masks.
        """
        if (frames is None) == (video_bytes is None):
            raise ValueError("pass exactly one of `frames` or `video_bytes`")

        if video_bytes is not None:
            frames = self._decode(video_bytes)
        frames = np.asarray(frames)
        if max_frames is not None:
            frames = frames[:max_frames]

        print(f"segmenting {len(frames)} frames, prompts={prompts}")

        session = self.processor.init_video_session(
            video=frames,
            inference_device="cuda",
            processing_device="cpu",
            video_storage_device="cpu",
        )
        self.processor.add_text_prompt(session, prompts)

        per_frame = {}
        # Omitting max_frame_num_to_track tracks every frame in the session; `frames` is
        # already sliced to max_frames above.
        for model_outputs in self.model.propagate_in_video_iterator(
            inference_session=session,
        ):
            out = self.processor.postprocess_outputs(session, model_outputs)
            per_frame[model_outputs.frame_idx] = self._summarise(out, return_masks)

        torch.cuda.empty_cache()
        return {
            "prompts": prompts,
            "num_frames": len(frames),
            "frame_shape": list(frames.shape[1:]),
            "frames": per_frame,
        }

    def _summarise(self, out: dict, return_masks: bool) -> dict:
        """Reduce one frame's masks to the scalars the geometric track actually consumes."""
        object_ids = out["object_ids"].tolist()
        scores = out["scores"].tolist()
        boxes = out["boxes"].cpu().numpy()
        masks = out["masks"].cpu().numpy().astype(bool)

        objects = []
        for i, obj_id in enumerate(object_ids):
            mask = masks[i]
            area = int(mask.sum())
            # Centroid is the mask-level stand-in for "where is the object" — this is what
            # feeds hand-object distance and the gravity-aligned velocity component.
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
            # Which prompt found what — needed when tracking hand and object together.
            "prompt_to_obj_ids": {
                k: [int(v) for v in vs]
                for k, vs in out.get("prompt_to_obj_ids", {}).items()
            },
        }


@app.local_entrypoint()
def main(
    videos: str = "clips",
    prompts: str = "person",
    max_frames: int = 0,
    out: str = "sam3_out",
    return_masks: bool = False,
):
    """
    Batch a directory of clips through one warm SAM 3 container.

        modal run sam3/app.py --videos clips --prompts "the towel,hand"

    No deploy needed — `modal run` builds the image, runs this entrypoint locally, and tears
    the app down when it returns. The model loads once per container (@modal.enter), not once
    per clip, so ten clips through one container means one load.

    max_frames defaults to 0, meaning every frame. Set it to cap long episodes while
    iterating — masks in particular grow linearly with frame count.
    """
    import json
    from pathlib import Path

    video_paths = sorted(
        p for p in Path(videos).iterdir() if p.suffix.lower() in {".mp4", ".mov", ".avi"}
    )
    if not video_paths:
        raise SystemExit(f"no videos found in {videos}/")

    prompt_list = [p.strip() for p in prompts.split(",") if p.strip()]
    print(f"{len(video_paths)} clips, prompts={prompt_list}, max_containers={MAX_CONTAINERS}")

    blobs = [p.read_bytes() for p in video_paths]
    segmenter = ConceptSegmenter()

    out_dir = Path(out)
    out_dir.mkdir(exist_ok=True)

    # One input iterator per argument: prompts is held constant across clips, video_bytes varies.
    # return_exceptions keeps one bad clip from killing the whole batch — with real episodes
    # you want the other nine results, not a traceback.
    results = segmenter.segment.map(
        [prompt_list] * len(blobs),
        blobs,
        # 0 -> None -> no cap, every frame in the clip.
        kwargs={"max_frames": max_frames or None, "return_masks": return_masks},
        return_exceptions=True,
    )

    for path, result in zip(video_paths, results):
        if isinstance(result, Exception):
            print(f"  {path.name}: FAILED — {result}")
            continue
        tracks = {o["object_id"] for f in result["frames"].values() for o in f["objects"]}
        note = ""
        if return_masks:
            note = f", masks -> {_write_masks(out_dir, path.stem, result)}"
        print(f"  {path.name}: {result['num_frames']} frames, {len(tracks)} tracks{note}")
        (out_dir / f"{path.stem}.json").write_text(json.dumps(result))

    print(f"✓ traces written to {out_dir}/")


def _write_masks(out_dir, stem, result) -> str:
    """
    Split bit-packed masks out of the trace into a flat .bin sidecar.

    Raw `bytes` are not JSON-serialisable, and full-resolution masks would bloat the trace
    past the point of being readable anyway. Each object's JSON record keeps
    `mask_offset` / `mask_nbytes` / `mask_shape`, so the two stay joinable with nothing but
    a seek. Deliberately stdlib-only — this runs on your machine, not in the container.
    """
    path = out_dir / f"{stem}.masks.bin"
    offset = 0
    with open(path, "wb") as fh:
        for frame_idx, frame in result["frames"].items():
            for obj in frame["objects"]:
                packed = obj.pop("mask_packed", None)
                if packed is None:
                    continue
                fh.write(packed)
                obj["mask_offset"] = offset
                obj["mask_nbytes"] = len(packed)
                offset += len(packed)
    return f"{path.name} ({offset / 1e6:.1f} MB)"
