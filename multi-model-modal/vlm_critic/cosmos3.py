"""
VLM critic, Cosmos 3 edition — same p(task complete) trace, different model lineage.

    modal run vlm_critic/cosmos3.py --limit 2 --max-frames 10
    modal run vlm_critic/cosmos3.py --stride 30

Cosmos3-Nano (16B, Mixture-of-Transformers) is an omnimodel: a diffusion tower for
generation plus an autoregressive REASONER tower for understanding. The reasoner is what
we want, and NVIDIA serves it through vLLM with an architecture override:

    vllm serve nvidia/Cosmos3-Nano \\
        --hf-overrides '{"architectures": ["Cosmos3ReasonerForConditionalGeneration"]}'

That means no transformers class to call for logits. Instead the OpenAI-compatible chat API
is asked for `logprobs` on a single generated token, and p(Yes) is read out of the returned
top-k distribution — the same TOPReward trick as vlm_critic/app.py, one layer up.

Why bother when Cosmos Reason 2 already exists in app.py: Reason 2 is a Qwen3-VL fine-tune,
so app.py's qwen-vs-cosmos comparison isolates *fine-tuning*. Cosmos3 is a different base
entirely, so this isolates *lineage*. If all three agree, the signal is real; if Cosmos3
alone wins, that says something about world-model pretraining. Ungated (OpenMDW 1.1).

Kept as a separate app because vLLM pins its own torch and would fight the transformers
image in app.py. Output layout matches: vlm_critic_out/cosmos3/<episode>.json + _meter.mp4.
"""

import json
from typing import Optional

import modal

app = modal.App(name="vlm-critic-cosmos3")

MODEL_ID = "nvidia/Cosmos3-Nano"
MODEL_KEY = "cosmos3"
CACHE_DIR = "/cache"
EPISODES_DIR = "/episodes"
OUTPUTS_DIR = "/outputs"
VLLM_PORT = 8000
MAX_CONTAINERS = 2

cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
vllm_cache_volume = modal.Volume.from_name("vllm-cache", create_if_missing=True)
# Which Volume holds the episodes. Override per run, e.g. EPISODES_VOLUME=egoverse-cup50
# (read at import time; `modal run` imports this file locally, so a shell env var works).
import os as _os
EPISODES_VOLUME = _os.environ.get("EPISODES_VOLUME", "egoverse-episodes")
episodes_volume = modal.Volume.from_name(EPISODES_VOLUME, create_if_missing=True)
outputs_volume = modal.Volume.from_name("egoverse-outputs", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])

# Modal's canonical vLLM image: CUDA devel base + vllm from PyPI. cosmos3_omni support is
# recent, so this needs a current vllm; if `Cosmos3ReasonerForConditionalGeneration` is
# unknown at startup, check the model card for the minimum vllm version and bump.
image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.27.1",
        "openai>=1.60",
        "numpy>=2.0,<3",
        "pillow>=11.0.0",
        "av>=15.0.0",
        "zarr>=3.0.0",
        "numcodecs>=0.13.0",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HUB_CACHE": CACHE_DIR})
)

with image.imports():
    import base64
    import io
    import math

    import numpy as np
    import zarr
    from PIL import Image as PILImage

QUESTION = (
    "Task: {task}\n"
    "Look at the current state of the scene. "
    "Has the task been completed successfully? Answer Yes or No."
)


def _episodes_root():
    """Directory that directly contains the episode folders (accepts nested layout)."""
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
    out = []
    for path in sorted(_episodes_root().iterdir()):
        meta = path / "zarr.json"
        if not meta.is_file():
            continue
        attrs = json.loads(meta.read_text()).get("attributes", {})
        out.append({"episode": path.name, "total_frames": attrs.get("total_frames")})
    return out


@app.cls(
    image=image,
    # 16B in bf16 is ~32GB of weights before KV cache and the vision tower.
    gpu="H100",
    timeout=3600,
    volumes={
        CACHE_DIR: cache_volume,
        "/root/.cache/vllm": vllm_cache_volume,
        EPISODES_DIR: episodes_volume,
        OUTPUTS_DIR: outputs_volume,
    },
    secrets=[hf_secret],
    scaledown_window=600,
    max_containers=MAX_CONTAINERS,
)
class Cosmos3Critic:
    @modal.enter()
    def start_server(self):
        """Bring up vLLM in-container and block until it answers /health."""
        import subprocess
        import time
        import urllib.request

        from openai import OpenAI

        cmd = [
            "vllm", "serve", MODEL_ID,
            "--hf-overrides", json.dumps({"architectures": ["Cosmos3ReasonerForConditionalGeneration"]}),
            "--tensor-parallel-size", "1",
            "--mm-encoder-tp-mode", "data",
            "--async-scheduling",
            "--max-logprobs", "20",
            "--port", str(VLLM_PORT),
            "--served-model-name", MODEL_KEY,
        ]
        print("starting:", " ".join(cmd))
        self.proc = subprocess.Popen(cmd)

        deadline = time.time() + 20 * 60  # first start downloads ~32GB
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"vllm exited early with code {self.proc.returncode}")
            try:
                urllib.request.urlopen(f"http://localhost:{VLLM_PORT}/health", timeout=2)
                break
            except Exception:
                time.sleep(3)
        else:
            raise RuntimeError("vllm did not become healthy in time")

        self.client = OpenAI(base_url=f"http://localhost:{VLLM_PORT}/v1", api_key="unused")
        print("vllm ready")

    @modal.exit()
    def stop_server(self):
        if getattr(self, "proc", None) and self.proc.poll() is None:
            self.proc.terminate()

    def _read_frames(self, episode: str, camera: str, stride: int, max_frames: Optional[int]):
        root = zarr.open_group(str(_episodes_root() / episode), mode="r")
        attrs = dict(root.attrs)
        arr = root[camera]
        n_stored = int(arr.shape[0])
        n = min(int(attrs.get("total_frames", n_stored)), n_stored)
        source_indices = list(range(0, n, max(1, stride)))
        if max_frames:
            source_indices = source_indices[:max_frames]
        if not source_indices:
            raise ValueError(f"{episode}: no frames to read (total_frames={n})")
        blobs = arr[source_indices]  # fancy-index once: plain bytes per element
        return list(blobs), source_indices, attrs

    def _p_yes(self, jpeg: bytes, task: str) -> tuple:
        """
        One chat call, max_tokens=1, logprobs on. Returns (p_yes, resolved) where resolved
        is False if neither Yes nor No appeared in the top-20 — then p_yes is 0.5 and the
        frame should be treated as uninformative rather than as evidence.
        """
        data_url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
        resp = self.client.chat.completions.create(
            model=MODEL_KEY,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": QUESTION.format(task=task)},
                ],
            }],
            max_tokens=1,
            temperature=0.0,
            logprobs=True,
            top_logprobs=20,
        )
        content = resp.choices[0].logprobs.content
        if not content:
            return 0.5, False
        p_yes = p_no = 0.0
        for item in content[0].top_logprobs:
            tok = item.token.strip().lower()
            if tok == "yes":
                p_yes += math.exp(item.logprob)
            elif tok == "no":
                p_no += math.exp(item.logprob)
        total = p_yes + p_no
        if total < 1e-9:
            return 0.5, False
        return p_yes / total, True

    @modal.method()
    def score_episode(
        self,
        episode: str,
        camera: str = "images.front_1",
        stride: int = 30,
        max_frames: Optional[int] = None,
        task_override: Optional[str] = None,
        render: bool = True,
    ) -> dict:
        blobs, source_indices, attrs = self._read_frames(episode, camera, stride, max_frames)
        task = task_override or attrs.get("task_description") or attrs.get("task_name") or ""
        print(f"{episode}: {len(blobs)} frames (stride {stride})")

        scored = [self._p_yes(b, task) for b in blobs]
        trace = [p for p, _ in scored]
        unresolved = sum(1 for _, ok in scored if not ok)

        task_name = attrs.get("task_name") or ""
        label = (
            "success" if task_name.endswith("_success")
            else "failure" if task_name.endswith("_failure")
            else None
        )

        meter_mp4 = None
        if render:
            fps = float(attrs.get("fps") or 30.0)
            frames = [
                _draw_meter(PILImage.open(io.BytesIO(b)).convert("RGB"), trace, k,
                            source_indices[k], label, MODEL_ID)
                for k, b in enumerate(blobs)
            ]
            meter_mp4 = _encode_mp4(frames, max(1, round(fps / max(1, stride))))
            import os
            out_path = f"{OUTPUTS_DIR}/vlm_critic/{MODEL_KEY}/{episode}_meter.mp4"
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "wb") as fh:
                fh.write(meter_mp4)
            outputs_volume.commit()

        return {
            "meter_mp4": meter_mp4,
            "episode": episode,
            "model": MODEL_ID,
            "model_key": MODEL_KEY,
            "task": task,
            "task_name": task_name,
            "label": label,
            "stride": stride,
            "source_indices": source_indices,
            "p_yes": trace,
            "unresolved_frames": unresolved,
            "p_yes_final": trace[-1] if trace else None,
            "p_yes_max": max(trace) if trace else None,
            "p_yes_last_quartile_mean": (
                float(np.mean(trace[-max(1, len(trace) // 4):])) if trace else None
            ),
        }


def _draw_meter(img, trace: list, k: int, source_idx: int, label, model_id: str):
    """Frame + p(done) bar + trace-so-far sparkline. Same layout as vlm_critic/app.py."""
    from PIL import ImageDraw

    W, H = img.size
    strip = 64
    canvas = PILImage.new("RGB", (W, H + strip), (18, 18, 20))
    canvas.paste(img, (0, 0))
    d = ImageDraw.Draw(canvas)
    p = float(trace[k])
    bx0, by0, bx1, by1 = 8, H + 8, W - 8, H + 24
    d.rectangle([bx0, by0, bx1, by1], outline=(90, 90, 95), width=1)
    fill = (56, 176, 0) if p >= 0.5 else (255, 89, 94)
    d.rectangle([bx0 + 1, by0 + 1, bx0 + 1 + int((bx1 - bx0 - 2) * p), by1 - 1], fill=fill)
    d.text((bx0, H + 28), f"p(done)={p:.2f}  frame {source_idx}  label={label}  {model_id.split('/')[-1]}",
           fill=(230, 230, 230))
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


@app.local_entrypoint()
def main(
    camera: str = "images.front_1",
    stride: int = 30,
    max_frames: int = 0,
    limit: int = 0,
    out: str = "vlm_critic_out",
    render: bool = True,
):
    """
    Score every episode with the Cosmos3-Nano reasoner via vLLM.

        modal run vlm_critic/cosmos3.py --limit 2 --max-frames 10   # shakeout
        modal run vlm_critic/cosmos3.py --stride 30                 # full slice

    First container start downloads ~32GB and compiles the engine — budget 10-15 minutes
    before the first frame is scored. Warm containers reuse both caches.
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
    names = [e["episode"] for e in episodes]
    print(f"{len(names)} episodes through {MODEL_ID} (vLLM), stride={stride}")

    out_dir = Path(out) / MODEL_KEY
    out_dir.mkdir(parents=True, exist_ok=True)

    results = Cosmos3Critic().score_episode.map(
        names,
        kwargs={"camera": camera, "stride": stride, "max_frames": max_frames or None, "render": render},
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
        unres = f"  ({result['unresolved_frames']} unresolved)" if result["unresolved_frames"] else ""
        print(
            f"  {name}: label={str(result['label']):<8} "
            f"p_yes_final={result['p_yes_final']:.3f} p_yes_max={result['p_yes_max']:.3f}{unres}"
        )

    labelled = [s for s in scored if s["label"]]
    for key in ("p_yes_final", "p_yes_last_quartile_mean"):
        succ = [s[key] for s in labelled if s["label"] == "success"]
        fail = [s[key] for s in labelled if s["label"] == "failure"]
        if succ and fail:
            print(f"\n{key}: success mean {sum(succ)/len(succ):.3f} (n={len(succ)})  "
                  f"failure mean {sum(fail)/len(fail):.3f} (n={len(fail)})  "
                  f"separation {sum(succ)/len(succ) - sum(fail)/len(fail):+.3f}")
    print(f"\n✓ {len(scored)}/{len(names)} traces written to {out_dir}/")
    if render and scored:
        print(f"  meters also in Modal storage: modal volume ls egoverse-outputs /vlm_critic/{MODEL_KEY}")
