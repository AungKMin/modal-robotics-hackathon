# SAM 3 — Promptable Concept Segmentation

Text-prompted instance segmentation and tracking over an episode. Give it a noun phrase
("the towel") and it returns masks plus stable object IDs for every matching instance,
across all frames.

## Model

[`facebook/sam3`](https://huggingface.co/facebook/sam3) — 860M params, a SAM 3 detector plus a
SAM 2-style memory tracker sharing one backbone. Runs on an L40S, which is also the GPU
[Modal's docs recommend](https://modal.com/docs/guide/gpu) as the inference default.

The official repo is **gated**. This app defaults to
[`jetjodh/sam3`](https://huggingface.co/jetjodh/sam3), an ungated mirror of the same
`Sam3VideoModel` weights, so it deploys with no token setup — the same move the
`text_to_image` template makes with SD 3.5. See the comment block in `app.py` to switch to
the official weights via a Modal Secret.

`sam3_video` only exists in **transformers 5.x**. The `transformers~=4.44` pin copied from
`text_to_image` will fail to import `Sam3VideoModel`.

## Run a batch (no deploy needed)

```bash
modal run sam3/app.py --videos clips --prompts "the towel,hand"
```

Builds the image, runs the batch, tears the app down when it returns. The model loads once
per container (`@modal.enter`), not once per clip — ten clips through one container is one
load. `MAX_CONTAINERS` in `app.py` sets the wall-clock/cold-start trade: 1 means a single
warm container processing clips sequentially, 10 means all at once but ten model loads.

First run pulls ~3.4GB of weights into the shared `hf-hub-cache` volume; warm after that.

## Deploy (for repeated calls)

```bash
modal deploy -m sam3.app
```

Worth it once you're calling from elsewhere or iterating on client code — deployed containers
stay warm between invocations for `scaledown_window` seconds, so you stop re-paying the model
load that `modal run` pays on every invocation.

```bash
# Smoke test against a sample clip
python sam3/try.py

# A real episode, prompted with the object noun from its language annotation
python sam3/try.py --video episode.mp4 --prompt "the towel" --prompt "hand" --max-frames 200
```

## Visualizing masks

Masks are off by default, so re-run with them on, then render:

```bash
modal run sam3/app.py --videos clips --prompts "person" --max-frames 20 --return-masks
uv pip install numpy pillow          # local only; the Modal image already has them
python3 sam3/visualize.py --all
```

Writes `sam3_out/<clip>_overlay.png` — a contact sheet of sampled frames with each mask
tinted, boxed, and labelled `id:score`. Colour is keyed to the SAM 3 object id, so a track
holds its colour across panels; a colour that jumps means identity was lost.

Run against a trace written *without* `--return-masks` and it says so, then draws boxes only.

## Output

`segment()` returns per-frame, per-object records — `object_id`, `score`, `box_xyxy`,
`area`, `centroid_xy` — plus `prompt_to_obj_ids` mapping each prompt to the tracks it found.

Masks are summarised rather than returned whole by default: a 500-frame episode at 1008px is
~500MB of bool mask per object, and the geometric features (hand–object distance, gravity-aligned
object velocity, "object left the frame") only need centroid, area, and box.

`--return-masks` adds bit-packed full-resolution masks, written to a `<clip>.masks.bin`
sidecar rather than inlined — raw bytes aren't JSON-serialisable, and inlining them would
make the trace unreadable. Each object record then carries `mask_offset` / `mask_nbytes` /
`mask_shape`, so recovering a mask is a seek plus `np.unpackbits`.

## Where this fits

Stage [A] of the pipeline. Mask-level contact replaces keypoint proximity in the hand–object
distance feature, and mask area collapsing or the centroid exiting the frame is a directly
measurable drop signal.

Prompts come from parsing the object noun out of each episode's language annotation — that
automation is the entire reason the plan specifies SAM 3 over SAM 2, since SAM 2 would need a
human click on frame one of every episode and the prevalence audit could not run at scale.
