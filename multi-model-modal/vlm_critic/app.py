"""
VLM critic — per-frame p(task complete) read from token probabilities.

    modal run vlm_critic/app.py --limit 2 --stride 30
    modal run vlm_critic/app.py --model cosmos --stride 15

This is deliverables #1 and #2 in one pass. For each sampled frame we ask the model
"Has the task been completed?" and read p("Yes") out of the *logits*, never the generated
text. The curve of p(yes) over time IS the confidence meter; where it crosses is the
failure onset; its aggregate is the episode verdict.

Why logits and not text: a generated "No" tells you the argmax and nothing else. p(yes)=0.49
and p(yes)=0.02 both print "No" but mean completely different things, and only the number
gives you a calibratable, thresholdable trace. This is the TOPReward trick.

It also means Instruct beats Thinking here: chain-of-thought tokens are pure cost when the
only thing read is a single distribution over the first answer token.

Both supported models share the qwen3_vl architecture — Cosmos Reason 2 is a fine-tune of
Qwen3-VL-8B-Instruct — so one code path serves both.
"""

import json
from typing import Optional

import modal

app = modal.App(name="vlm-critic")

CACHE_DIR = "/cache"
EPISODES_DIR = "/episodes"
MAX_CONTAINERS = 4

MODELS = {
    # apache-2.0, ungated. The default: nothing to accept, nothing to configure.
    "qwen": "Qwen/Qwen3-VL-8B-Instruct",
    # GATED — accept terms at https://huggingface.co/nvidia/Cosmos-Reason2-8B first.
    # Purpose-built for physical/embodied reasoning, which is exactly the critic role, so
    # it is the one to beat. The attached HF_TOKEN secret authorises the download.
    "cosmos": "nvidia/Cosmos-Reason2-8B",
    # GATED (Gemma terms) — accept at https://huggingface.co/google/paligemma2-3b-mix-448.
    # Not a chat model: a SigLIP+Gemma2 VQA model that answers "answer en <q>" prompts with
    # a short word. 3B, cheap, and a genuinely different lineage from the two above.
    "paligemma": "google/paligemma2-3b-mix-448",
}
# Which models take a chat template (Qwen-family) vs PaliGemma's raw "answer en" prefix.
CHAT_MODELS = {"qwen", "cosmos"}

cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
# Which Volume holds the episodes. Override per run, e.g. EPISODES_VOLUME=egoverse-cup50
# (read at import time; `modal run` imports this file locally, so a shell env var works).
import os as _os
EPISODES_VOLUME = _os.environ.get("EPISODES_VOLUME", "egoverse-episodes")
episodes_volume = modal.Volume.from_name(EPISODES_VOLUME, create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])
# Rendered confidence-meter videos are written here as well as returned, so the demo
# artefacts live in Modal storage independent of whoever ran the job.
OUTPUTS_DIR = "/outputs"
outputs_volume = modal.Volume.from_name("egoverse-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.10.0",
        "torchvision==0.25.0",
        "transformers==5.15.0",
        "accelerate>=1.11.0",
        "numpy>=2.0,<3",
        "pillow>=11.0.0",
        "av>=15.0.0",
        "zarr>=3.0.0",
        "numcodecs>=0.13.0",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HUB_CACHE": CACHE_DIR})
)

with image.imports():
    import io

    import numpy as np
    import torch
    import zarr
    from PIL import Image as PILImage

    from transformers import (
        AutoModelForMultimodalLM,
        AutoProcessor,
        PaliGemmaForConditionalGeneration,
    )

# Ask about the GOAL STATE, not the process. A single frame cannot show whether a handover
# happened, so "has the task been completed?" is unanswerable from one image and the model
# says No everywhere (observed: p(yes) ~ 1e-4 on every frame of every episode). Whether the
# cup is on the saucer *is* visible in a frame. Override per task with --question.
QUESTION = (
    "Task: {task}\n"
    "Look only at what is visible in this image. Is the final goal state of the task "
    "achieved right now (the object is resting where the task says it should end up)? "
    "Answer Yes or No."
)



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
def list_episodes() -> list[dict]:
    """Enumerate episodes in the Volume with their metadata (reads only each root zarr.json)."""
    from pathlib import Path

    out = []
    for path in sorted(_episodes_root().iterdir()):
        meta = path / "zarr.json"
        if not meta.is_file():
            continue
        attrs = json.loads(meta.read_text()).get("attributes", {})
        out.append(
            {
                "episode": path.name,
                "task_name": attrs.get("task_name"),
                "task_description": attrs.get("task_description"),
                "total_frames": attrs.get("total_frames"),
            }
        )
    return out


@app.cls(
    image=image,
    # 8.8B params in bf16 is ~17.5GB of weights before activations and the vision tower.
    # A100-40GB has the headroom; L40S (48GB) also fits and is usually cheaper to get.
    gpu="A100-40GB",
    timeout=3600,
    volumes={CACHE_DIR: cache_volume, EPISODES_DIR: episodes_volume, OUTPUTS_DIR: outputs_volume},
    secrets=[hf_secret],
    scaledown_window=300,
    max_containers=MAX_CONTAINERS,
)
class Critic:
    model_key: str = modal.parameter(default="qwen")

    @modal.enter()
    def load_model(self):
        model_id = MODELS[self.model_key]
        print(f"loading {model_id}")
        self.processor = AutoProcessor.from_pretrained(model_id)
        if self.model_key in CHAT_MODELS:
            self.model = AutoModelForMultimodalLM.from_pretrained(
                model_id, dtype=torch.bfloat16, device_map="cuda"
            )
        else:
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                model_id, dtype=torch.bfloat16, device_map="cuda"
            )
        self.model.eval()

        tok = self.processor.tokenizer
        # Score the answer token in the exact form it would be generated in: after a
        # newline the model emits "Yes" with a leading space in most BPE vocabularies, so
        # collect every surface variant and sum their mass rather than betting on one id.
        self.yes_ids = _variant_ids(tok, ["Yes", "yes", "YES"])
        self.no_ids = _variant_ids(tok, ["No", "no", "NO"])
        print(f"yes ids: {sorted(self.yes_ids)}  no ids: {sorted(self.no_ids)}")
        if not self.yes_ids or not self.no_ids:
            raise RuntimeError("could not resolve Yes/No token ids for this tokenizer")

    def _read_frames(self, episode: str, camera: str, stride: int, max_frames: Optional[int]):
        root = zarr.open_group(str(_episodes_root() / episode), mode="r")
        attrs = dict(root.attrs)
        arr = root[camera]
        # zarr v3 Arrays have no len(); total_frames is authoritative over the padded shape.
        n_stored = int(arr.shape[0])
        n = min(int(attrs.get("total_frames", n_stored)), n_stored)
        source_indices = list(range(0, n, max(1, stride)))
        if max_frames:
            source_indices = source_indices[:max_frames]
        if not source_indices:
            raise ValueError(f"{episode}: no frames to read (total_frames={n})")
        # Fancy-index once: scalar arr[i] is a doubly wrapped object ndarray whose bytes()
        # is a pointer, not the JPEG. List indexing yields plain bytes in a single read.
        blobs = arr[source_indices]
        images = [PILImage.open(io.BytesIO(b)).convert("RGB") for b in blobs]
        return images, source_indices, attrs

    def _p_yes(self, image, task: str) -> float:
        """
        One forward pass; return p(Yes) renormalised over {Yes, No}.

        No generation at all — we only need the distribution over the first answer token,
        so `generate()` would be wasted work. Renormalising over just the two answers
        removes the mass the model puts on unrelated continuations, which otherwise makes
        the trace depend on unrelated formatting quirks.

        inference_mode is entered inside the body, not as a decorator: decorators evaluate
        at class-definition time, and `torch` only exists inside `image.imports()`, which
        is a no-op locally. `modal run` imports this module locally to discover the app, so
        a `@torch....` decorator would NameError before anything reached a container.
        """
        with torch.inference_mode():
            q = self._question.format(task=task)
            if self.model_key in CHAT_MODELS:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": q},
                        ],
                    }
                ]
                prompt = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.processor(
                    text=[prompt], images=[image], return_tensors="pt"
                ).to(self.model.device)
            else:
                # PaliGemma: the processor inserts the image tokens itself; the "answer en"
                # prefix is how the mix checkpoints were trained to do VQA. Keep the question
                # to one line — it is a short-answer model, not an instruction follower.
                short_q = q.split("\n")[-1].strip()
                inputs = self.processor(
                    images=image, text=f"answer en {short_q}", return_tensors="pt"
                ).to(self.model.device)

            logits = self.model(**inputs).logits[0, -1].float()
            probs = torch.softmax(logits, dim=-1)
            p_yes = probs[self.yes_ids].sum().item()
            p_no = probs[self.no_ids].sum().item()
            top = torch.topk(probs, 5)
            self._last_top = [
                (self.processor.tokenizer.decode([int(i)]), round(float(v), 4))
                for v, i in zip(top.values, top.indices)
            ]

        total = p_yes + p_no
        # Degenerate case: the model put essentially no mass on either answer. Report the
        # midpoint rather than a divide-by-zero or a fake confident number.
        return 0.5 if total < 1e-9 else p_yes / total

    @modal.method()
    def score_episode(
        self,
        episode: str,
        camera: str = "images.front_1",
        stride: int = 30,
        max_frames: Optional[int] = None,
        task_override: Optional[str] = None,
        render: bool = True,
        question: Optional[str] = None,
    ) -> dict:
        images, source_indices, attrs = self._read_frames(
            episode, camera, stride, max_frames
        )
        task = task_override or attrs.get("task_description") or attrs.get("task_name") or ""
        print(f"{episode}: {len(images)} frames (stride {stride})")

        self._question = question or QUESTION
        trace = [self._p_yes(img, task) for img in images]
        # What the model actually wanted to say on the LAST frame — the single most useful
        # diagnostic when a trace looks flat.
        last_top = getattr(self, "_last_top", None)

        task_name = attrs.get("task_name") or ""
        label = (
            "success" if task_name.endswith("_success")
            else "failure" if task_name.endswith("_failure")
            else None
        )

        # The confidence meter as a video: each sampled frame with p(done) drawn under it
        # and the trace-so-far as a sparkline. This IS deliverable #2 in demo form.
        meter_mp4 = None
        if render:
            fps = float(attrs.get("fps") or 30.0)
            frames = [
                _draw_meter(img, trace, k, source_indices[k], label, MODELS[self.model_key])
                for k, img in enumerate(images)
            ]
            meter_mp4 = _encode_mp4(frames, max(1, round(fps / max(1, stride))))
            import os
            out_path = f"{OUTPUTS_DIR}/vlm_critic/{self.model_key}/{episode}_meter.mp4"
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "wb") as fh:
                fh.write(meter_mp4)
            outputs_volume.commit()

        return {
            "meter_mp4": meter_mp4,
            "episode": episode,
            "model": MODELS[self.model_key],
            "model_key": self.model_key,
            "task": task,
            "task_name": task_name,
            "label": label,
            "stride": stride,
            "source_indices": source_indices,
            # The confidence meter (deliverable #2), one value per sampled frame.
            "p_yes": trace,
            "question": self._question,
            "last_frame_top_tokens": last_top,
            # Episode verdict (deliverable #1). Late frames carry the signal — a task is
            # judged by how it ended, not by its average over time, so summarising with a
            # mean would wash out exactly the evidence that matters.
            "p_yes_final": trace[-1] if trace else None,
            "p_yes_max": max(trace) if trace else None,
            "p_yes_last_quartile_mean": (
                float(np.mean(trace[-max(1, len(trace) // 4):])) if trace else None
            ),
        }



def _draw_meter(img, trace: list, k: int, source_idx: int, label, model_id: str):
    """
    Frame + confidence meter. A filled bar for p(done) at this frame, the trace so far as
    a sparkline, and the ground-truth label so a viewer can judge the curve at a glance.
    """
    from PIL import ImageDraw

    W, H = img.size
    strip = 64
    canvas = PILImage.new("RGB", (W, H + strip), (18, 18, 20))
    canvas.paste(img, (0, 0))
    d = ImageDraw.Draw(canvas)
    p = float(trace[k])
    # bar
    bx0, by0, bx1, by1 = 8, H + 8, W - 8, H + 24
    d.rectangle([bx0, by0, bx1, by1], outline=(90, 90, 95), width=1)
    fill = (56, 176, 0) if p >= 0.5 else (255, 89, 94)
    d.rectangle([bx0 + 1, by0 + 1, bx0 + 1 + int((bx1 - bx0 - 2) * p), by1 - 1], fill=fill)
    d.text((bx0, H + 28), f"p(done)={p:.2f}  frame {source_idx}  label={label}  {model_id.split('/')[-1]}",
           fill=(230, 230, 230))
    # sparkline of the trace so far, right-aligned in the strip
    n = len(trace)
    if n > 1:
        sx0, sy0, sw, sh = W - 8 - 160, H + 40, 160, 20
        pts = [(sx0 + int(sw * i / (n - 1)), sy0 + sh - int(sh * float(trace[i]))) for i in range(k + 1)]
        d.rectangle([sx0, sy0, sx0 + sw, sy0 + sh], outline=(60, 60, 65), width=1)
        d.line([(sx0, sy0 + sh // 2), (sx0 + sw, sy0 + sh // 2)], fill=(60, 60, 65), width=1)
        if len(pts) > 1:
            d.line(pts, fill=(255, 202, 58), width=2)
    return np.asarray(canvas)


def _encode_mp4(frames: list, fps: int) -> bytes:
    """In-memory H.264 mp4 via PyAV; mpeg4 fallback; even dims for yuv420p."""
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

def _variant_ids(tok, words: list[str]) -> list[int]:
    """
    Collect every single-token id that renders as one of `words`, with or without a leading
    space. Vocabularies differ on which variant a model actually emits, and guessing one
    id silently produces a trace of near-zero probabilities that looks like real data.
    """
    ids = set()
    for word in words:
        for surface in (word, f" {word}"):
            enc = tok.encode(surface, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(enc[0])
    return sorted(ids)


@app.local_entrypoint()
def main(
    model: str = "qwen",
    camera: str = "images.front_1",
    stride: int = 30,
    max_frames: int = 0,
    limit: int = 0,
    out: str = "vlm_critic_out",
    render: bool = True,
    question: str = "",
    match: str = "",
):
    """
    Score every episode in the Volume and write one p(yes) trace per episode.

        modal run vlm_critic/app.py --limit 2 --max-frames 10   # shakeout
        modal run vlm_critic/app.py --stride 30                 # full slice

    stride is the cost knob and it bites harder here than for SAM 3: this is one full VLM
    forward pass per frame, so stride 30 (one frame per second at 30fps) is a sane start.
    The 20-episode slice at stride 30 is ~700 forward passes.
    """
    from pathlib import Path

    if model not in MODELS:
        raise SystemExit(f"--model must be one of {sorted(MODELS)}")

    episodes = list_episodes.remote()
    if not episodes:
        raise SystemExit(
            "No episodes in the Volume. Upload them first:\n"
            "  modal volume put egoverse-episodes /home/asubuntu/egoverse_data"
        )
    if match:
        episodes = [e for e in episodes if e["episode"].startswith(match)]
    if limit:
        episodes = episodes[:limit]

    names = [e["episode"] for e in episodes]
    print(f"{len(names)} episodes through {MODELS[model]}, stride={stride}")

    out_dir = Path(out) / model   # one subfolder per model so qwen/cosmos never overwrite
    out_dir.mkdir(parents=True, exist_ok=True)

    results = Critic(model_key=model).score_episode.map(
        names,
        kwargs={"camera": camera, "stride": stride, "max_frames": max_frames or None,
                "render": render, "question": question or None},
        return_exceptions=True,
    )

    scored = []
    for name, result in zip(names, results):
        if isinstance(result, Exception):
            print(f"  {name}: FAILED — {result}")
            continue
        mp4 = result.pop("meter_mp4", None)
        if mp4:
            (out_dir / f"{name}_meter.mp4").write_bytes(mp4)
        (out_dir / f"{name}.json").write_text(json.dumps(result))
        scored.append(result)
        print(
            f"  {name}: label={result['label']:<8} "
            f"p_yes_final={result['p_yes_final']:.3f} "
            f"p_yes_max={result['p_yes_max']:.3f}"
        )

    _report(scored)
    print(f"✓ {len(scored)}/{len(names)} traces written to {out_dir}/")


def _report(scored: list[dict]) -> None:
    """
    Separation check against the labels already in the data.

    Not a substitute for the real eval (AUROC + reliability curve), but it answers the only
    question that matters before spending more GPU: does the signal separate the classes at
    all? If the two means overlap, the prompt is wrong and no amount of fusion will save it.
    """
    labelled = [s for s in scored if s["label"] and s["p_yes_final"] is not None]
    if not labelled:
        return
    for key in ("p_yes_final", "p_yes_last_quartile_mean"):
        succ = [s[key] for s in labelled if s["label"] == "success"]
        fail = [s[key] for s in labelled if s["label"] == "failure"]
        if succ and fail:
            gap = sum(succ) / len(succ) - sum(fail) / len(fail)
            print(
                f"\n{key}: success mean {sum(succ)/len(succ):.3f} (n={len(succ)})  "
                f"failure mean {sum(fail)/len(fail):.3f} (n={len(fail)})  "
                f"separation {gap:+.3f}"
            )
