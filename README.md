# Human Reward Model — EgoVerse Track 3

A success/failure critic for manipulation demos, running on [Modal](https://modal.com). Given
an EgoVerse episode (robot **or** human egocentric) it outputs a per-episode
success/failure tag, a per-frame confidence trace, and a dataset-level prevalence audit —
with no new human labelling and no model training.

Built in one day for the EgoVerse Data Optimization & Evaluation Suite hackathon.

https://github.com/user-attachments/assets/bf97fde3-0637-4622-8e00-3abb104e7c60

## The result: a failure audit of 50 human cup-on-saucer demos

> One-slide version: [`SLIDE.pptx`](SLIDE.pptx) · [`SLIDE.png`](SLIDE.png) · [`SLIDE.html`](SLIDE.html) · [`SLIDE.md`](SLIDE.md)

**Success = the cup ends up on the saucer; failure = it does not.** No labels exist for these
50 clips ([somundane/egoverse-cup50](https://huggingface.co/datasets/somundane/egoverse-cup50)),
so this is the Track 3 prevalence audit proper: four independent signals, every one run on
every clip, fused into one verdict per episode.

| signal | how | tagged success | failure prevalence |
|---|---|---|---|
| **SAM 3** geometry | text-prompted masks for `cup`, `saucer`; is the cup centroid inside the saucer box at the end? | 36/50 | 28% |
| **Qwen3-VL-8B** | p("Yes") from logits to *"is the cup resting on the saucer?"*, last quarter of frames, 3 fps | 33/50 | 34% |
| **PaliGemma2-3B** | same question, VQA-style prompt | 41/50 | 18% |
| **Cosmos-Reason2-8B** | same question, embodied-reasoning fine-tune of Qwen3-VL | 37/50 | 26% |
| **Fused** = mean(SEG, mean of VLMs) ≥ 0.5 | geometry and language weighted equally | **34/50** | **32%** |

**Agreement:** 26/50 clips unanimous across all four signals, 37/50 with ≥3 of 4, 6 split 2–2.
Per-episode votes: [`results/cup50_final.md`](results/cup50_final.md). Two independent kinds of
evidence — geometry and language — land on the same prevalence (28% vs 32%).

**Robustness to sampling rate:** the same audit at 1 fps (7 frames/clip from the HF parquet
instead of 20 from the 30 fps source) gives 35/15, 30% — **1 verdict out of 50 differs**, 27/50
unanimous. Individual models move more (Qwen 39→33 tagged success, Cosmos 34→37); the fusion
absorbs it. [`results/cup50_1fps_final.md`](results/cup50_1fps_final.md).

<table><tr>
<td><img src="demo/gifs/sam3_cup50_10fps_success.gif" width="440"><br><sub>SAM 3 at 10 fps: <code>hand</code>, <code>cup</code>, <code>saucer</code> from text prompts — a unanimous success (4/4)</sub></td>
<td><img src="demo/gifs/meter_cup50_qwen.gif" width="440"><br><sub>The confidence meter: p(cup on saucer) per frame from Qwen3-VL logits, same clip</sub></td>
</tr></table>

<table><tr>
<td><img src="demo/stills/unanimous_success_692e98927641010d04354574_end.jpg" width="300"><br><sub>✅ unanimous success — SEG 1.00 · VLMs 0.87/0.81/0.96</sub></td>
<td><img src="demo/stills/unanimous_failure_2026-01-11-23-11-22-998000_end.jpg" width="300"><br><sub>❌ unanimous failure — SEG 0.00 · VLMs 0.00/0.07/0.00</sub></td>
<td><img src="demo/stills/disputed_seg_yes_vlm_split_692ea3beffdc0ca6345c4246_end.jpg" width="300"><br><sub>SEG 1.00, VLMs 0.74/0.46/0.63 → ✅ 3/4 (was 2/4 at 1 fps: denser sampling brought the VLMs to the geometry)</sub></td>
<td><img src="demo/stills/disputed_seg_no_vlm_yes_692ea671dbc4294a49cc727e_end.jpg" width="300"><br><sub>SEG 0.00, VLMs 0.32/0.86/0.04 → ❌ 1/4 (was 3 VLM yes-votes at 1 fps)</sub></td>
</tr></table>

<table><tr>
<td><img src="results/prevalence.png" width="560"><br><sub>Tags per dataset × model at a fixed 0.5</sub></td>
<td><img src="results/agreement_cup50.png" width="300"><br><sub>Pairwise tag agreement on cup50</sub></td>
</tr></table>

Every trace carries `source_indices` (sampled position → original frame), the model's top-5 first
tokens on the last frame (so you can see it is answering Yes/No and not something else), and the
overlay / meter videos are rendered in-container and written to both a local `<model>_out/` folder
and the `egoverse-outputs` Modal Volume. More clips, meters and stills: [`demo/`](demo/README.md).

## Side experiments

### Validation on a labelled robot slice
20 `eva_bimanual` cup-on-saucer episodes, 10 success / 10 failure — the only labels we had, used to
check the signals are real before running them on human video.

| signal | accuracy | AUROC |
|---|---|---|
| VLM p(done), last quarter (Qwen3-VL-8B) | 0.70 | **0.83** |
| SAM 3 cup-centroid-in-saucer | 0.70 | 0.64 |
| Fused + geometric veto | 0.70 | 0.74 |

<table><tr>
<td><img src="results/eva_roc.png" width="330"></td>
<td><img src="demo/gifs/meter_eva_failure_qwen.gif" width="440"><br><sub>A labelled failure: the meter never rises</sub></td>
</tr></table>

Robot episodes also carry gripper proprioception, so a third signal exists there: `obs_gripper` vs
`cmd_gripper` gives holds, releases and slips directly (`geometric/`). Finding: **zero gripper slips
across all 10 failures** — on this task failures are hesitation, not drops.

### One 2-minute egocentric video, split into attempts
An aria head-cam episode is not one demo — the person places the cup 25 times. Segmenting on cup
rests *relative to the saucer* (so head motion cancels) and scoring each pick-up→set-down:
**25 attempts → 16 success / 9 failure**, plus 16 returns to the start position, correctly not counted.

<table><tr>
<td><img src="demo/gifs/sam3_aria_hands.gif" width="420"><br><sub>SAM 3 tracking both hands, cup and saucer on the head-cam</sub></td>
<td><img src="results/aria_split_2025-11-24-23-59-28-546000.png" width="560"><br><sub>cup speed rel. saucer · cup-in-saucer · VLM p(done), attempts shaded</sub></td>
</tr></table>

Table: [`results/aria_split_2025-11-24-23-59-28-546000.md`](results/aria_split_2025-11-24-23-59-28-546000.md).

### Three things we learned (and can prove)

1. **Ask about the goal *state*, not "was the task completed?"** The latter is unanswerable from
   one frame; p(yes) sat at 10⁻⁴ on every frame while the *ranking* still worked (AUROC 0.73 →
   0.83 after the fix). And the smaller the model, the more literal the question: PaliGemma
   hedged at 0.28–0.65 on the generic prompt and became decisive (0.07–0.97) on "is the cup
   resting on the saucer?". Only visible because we read logits — 0.49 and 0.02 both print "No".
2. **Robot failures here are not drops.** Zero gripper slips across all 10 failures; hesitation
   (off-hand speed, dominant-arm path length) separates the classes. Geometry flags, VLM adjudicates.
3. **The unit of evaluation must match the video.** A cup50 clip is one attempt — score its end.
   An aria episode is 25 attempts — segment first, then score each. Camera motion has to cancel to
   segment: cup speed relative to a static scene object, not in image space.

**Cost:** geometric track is CPU, seconds. VLM at 1 fps is one batched forward pass per frame
(16/pass, H100); all 50 clips × 3 models ran in minutes. A full-dataset sweep is a `.map()`.

Full report with every graph and per-episode verdict: [`results/`](results/README.md).

## Run it

**One episode, the whole critic** — SAM 3 + all three VLMs in parallel, fused into a verdict card:

```bash
cd multi-model-modal
python3 run_episode.py 692e98927641010d04354574                    # a cup50 clip at 3 fps
python3 run_episode.py 2025-11-24-23-59-28 --volume egoverse-aria --stride 15
```
Writes `run_episode_out/<episode>/verdict.md` plus the traces, overlay and meter videos, and each
model's top tokens. Legs that fail transiently are retried once; logs are kept next to the verdict.

**Everything else:**

```bash
modal setup                                   # once
modal secret create huggingface-secret HF_TOKEN=hf_...   # gated weights (PaliGemma, Cosmos)
cd multi-model-modal

# data → Modal Volumes (once)
modal volume create egoverse-episodes && modal volume put egoverse-episodes /path/to/eva_zarr
modal run cup50/prepare.py                    # HF parquet → Volume egoverse-cup50

# the three signals, then fusion
modal run sam3/episodes.py --stride 5 --prompts "hand,cup,saucer"
modal run vlm_critic/app.py --model qwen --stride 30       # or paligemma / cosmos
modal run geometric/app.py                                 # CPU, seconds
python3 fuse/fuse.py                                       # → fuse_out/summary.md

# any other Volume of episodes, e.g. the human sets
EPISODES_VOLUME=egoverse-cup50 modal run vlm_critic/app.py --stride 1 --out vlm_critic_out_cup50
```

Each folder's README documents its flags, outputs and the design decisions behind it.
`CLAUDE.md` records the data conventions that were verified along the way.

## Repo layout

```
multi-model-modal/
  sam3/         app.py (clips) · episodes.py (zarr episodes, overlay video) · visualize.py
  vlm_critic/   app.py (Qwen3-VL / Cosmos-Reason2 / PaliGemma) · cosmos3.py (Cosmos3 via vLLM)
  geometric/    proprioception features, CPU
  fuse/         fuse.py — tags, accuracy/AUROC, prevalence · report.py — results/ tables + graphs
  cup50/        prepare.py — HF parquet → zarr Volume
results/        cross-model report (generated by fuse/report.py)
demo/           sample overlays, confidence meters, result summaries
sync_s3.py      EgoVerse S3 sync with named episode presets
CLAUDE.md       data conventions verified along the way (gripper polarity, frames, zarr quirks)
```

## Method

**Three signals, one fusion step.**

- **VLM critic** (`vlm_critic/`) — for each sampled frame, ask "is the goal state visible?" and
  read `p(Yes)` from the first-token logits, renormalised over {Yes, No}. The per-frame curve
  is the confidence meter; its late-window mean (or peak, for long egocentric video) is the
  episode score. Instruct models only: chain-of-thought is wasted tokens when only the first
  distribution is read. Three lineages served through one code path so the comparison is
  apples-to-apples: Qwen3-VL-8B, Cosmos-Reason2-8B (a Qwen3-VL fine-tune — isolates
  fine-tuning), PaliGemma 2 (SigLIP+Gemma — isolates lineage).
- **SAM 3** (`sam3/`) — Promptable Concept Segmentation: the object nouns from the task
  description are the prompts, so no per-episode human seeding, which is what makes a dataset
  sweep possible. Yields per-frame masks and stable track ids for `hand`, `cup`, `saucer`; the
  cup-centroid-in-saucer-box test over the last quarter is the geometric success criterion.
  Overlays are rendered in-container at full frame rate with held masks.
- **Geometric track** (`geometric/`) — robot episodes only. `obs_gripper` is a measured
  aperture, `cmd_gripper` the commanded one; a settled plateau above the closed-on-air band while
  commanded closed is a *hold*, ending with cmd still closed and obs back in the air band is a
  *slip*, both grippers holding within 30 cm in a shared camera frame is a *handover*. CPU only.
- **Fusion** (`fuse/`) — mean of the available continuous signals with a geometric veto
  (no hold → no success). Accuracy and AUROC where labels exist; prevalence where they don't.

## Conventions we verified rather than assumed

- Gripper `0 = closed, 1 = open` — checked on the wrist camera (fingers touching at 0.000).
- Each arm's `obs_ee_pose` is in its own base frame; `extrinsics[arm]` is the camera pose in that
  frame, so `inv(extrinsics)` puts both arms in one frame — verified by projecting through the
  intrinsics onto the front image.
- On a head-mounted camera the saucer box must be per-frame; an episode-median box tagged
  50/50 human episodes as failure.
- zarr v3 `Array` has no `len()`; scalar indexing returns a doubly wrapped object array whose
  `bytes()` is a pointer, not the JPEG. Fancy-index once.
- `modal volume put … /` nests the directory by name; apps resolve the episodes root either way.

## Limitations

- The labelled slice is 20 episodes of one task on one robot; the prevalence numbers on the
  human sets are from 8 VLM-scored episodes each (50 for the SAM 3 criterion on cup50).
- Thresholds are fixed at 0.5 and not tuned; AUROC is the number to trust.
- The scoring window (`late` vs `max`) is a per-dataset choice, stated in the report, not learned.
- Cosmos3-Nano's reasoner (via vLLM) is wired but was not run in the sprint.
- Dyn-HaMR / MANO hand pose was not used: the dev slice was a bimanual robot, and on the human
  sets SAM 3's `hand` track covered what the criteria needed. It is the natural next step for
  hand-pose-based failure signatures on egocentric video.

## Team

Four-person team, EgoVerse hackathon, Track 3.

## References

- EgoVerse dataset — https://github.com/GaTech-RL2/EgoVerse
- SAM 3: Segment Anything with Concepts — https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/
- Qwen3-VL — https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
- NVIDIA Cosmos Reason 2 — https://huggingface.co/nvidia/Cosmos-Reason2-8B
- PaliGemma 2 — https://huggingface.co/google/paligemma2-3b-mix-448
- TOPReward (progress from token probabilities) — https://arxiv.org/html/2602.19313
- Modal — https://modal.com
