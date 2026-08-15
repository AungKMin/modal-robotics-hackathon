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
}

cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
episodes_volume = modal.Volume.from_name("egoverse-episodes", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.10.0",
        "torchvision==0.25.0",
        "transformers==5.15.0",
        "accelerate>=1.11.0",
        "numpy>=2.0,<3",
        "pillow>=11.0.0",
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

    from transformers import AutoModelForMultimodalLM, AutoProcessor

QUESTION = (
    "Task: {task}\n"
    "Look at the current state of the scene. "
    "Has the task been completed successfully? Answer Yes or No."
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
    volumes={CACHE_DIR: cache_volume, EPISODES_DIR: episodes_volume},
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
        self.model = AutoModelForMultimodalLM.from_pretrained(
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
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": QUESTION.format(task=task)},
                    ],
                }
            ]
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[prompt], images=[image], return_tensors="pt"
            ).to(self.model.device)

            logits = self.model(**inputs).logits[0, -1].float()
            probs = torch.softmax(logits, dim=-1)
            p_yes = probs[self.yes_ids].sum().item()
            p_no = probs[self.no_ids].sum().item()

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
    ) -> dict:
        images, source_indices, attrs = self._read_frames(
            episode, camera, stride, max_frames
        )
        task = task_override or attrs.get("task_description") or attrs.get("task_name") or ""
        print(f"{episode}: {len(images)} frames (stride {stride})")

        trace = [self._p_yes(img, task) for img in images]

        task_name = attrs.get("task_name") or ""
        return {
            "episode": episode,
            "model": MODELS[self.model_key],
            "task": task,
            "task_name": task_name,
            "label": (
                "success" if task_name.endswith("_success")
                else "failure" if task_name.endswith("_failure")
                else None
            ),
            "stride": stride,
            "source_indices": source_indices,
            # The confidence meter (deliverable #2), one value per sampled frame.
            "p_yes": trace,
            # Episode verdict (deliverable #1). Late frames carry the signal — a task is
            # judged by how it ended, not by its average over time, so summarising with a
            # mean would wash out exactly the evidence that matters.
            "p_yes_final": trace[-1] if trace else None,
            "p_yes_max": max(trace) if trace else None,
            "p_yes_last_quartile_mean": (
                float(np.mean(trace[-max(1, len(trace) // 4):])) if trace else None
            ),
        }


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
    if limit:
        episodes = episodes[:limit]

    names = [e["episode"] for e in episodes]
    print(f"{len(names)} episodes through {MODELS[model]}, stride={stride}")

    out_dir = Path(out)
    out_dir.mkdir(exist_ok=True)

    results = Critic(model_key=model).score_episode.map(
        names,
        kwargs={"camera": camera, "stride": stride, "max_frames": max_frames or None},
        return_exceptions=True,
    )

    scored = []
    for name, result in zip(names, results):
        if isinstance(result, Exception):
            print(f"  {name}: FAILED — {result}")
            continue
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
