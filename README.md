# Human Reward Model — EgoVerse Track 3

A success/failure critic for manipulation demos, running on [Modal](https://modal.com). Given
an EgoVerse episode (robot **or** human egocentric) it outputs a per-episode
success/failure tag, a per-frame confidence trace, and a dataset-level prevalence audit —
with no new human labelling and no model training.

Built in one day for the EgoVerse Data Optimization & Evaluation Suite hackathon.

https://github.com/user-attachments/assets/bf97fde3-0637-4622-8e00-3abb104e7c60

## Results at a glance

> One-slide version: [`SLIDE.pptx`](SLIDE.pptx) · [`SLIDE.png`](SLIDE.png) · [`SLIDE.html`](SLIDE.html) · [`SLIDE.md`](SLIDE.md)

**Does the demo succeed? Ask the VLM's logits, not its words.**
Per-frame `p(cup is on the saucer)` read from Qwen3-VL-8B / Cosmos-Reason2-8B / PaliGemma 2 logits →
confidence meter → episode tag → prevalence audit. Fused with SAM 3 text-prompted masks
(`hand`, `cup`, `saucer`) and, on robot episodes, gripper proprioception. Zero training, zero new
labels, all on Modal. **Success = the cup ends up on the saucer; failure = it does not.**

**Labelled robot slice** — 20 eva cup-on-saucer episodes, 10 success / 10 failure

| signal | accuracy | AUROC |
|---|---|---|
| VLM p(done), last quarter (Qwen3-VL-8B) | 0.70 | **0.83** |
| SAM 3 cup-centroid-in-saucer | 0.70 | 0.64 |
| Fused + geometric veto | 0.70 | 0.74 |

**Prevalence audit — 50 unlabelled human cup-on-saucer clips**
([somundane/egoverse-cup50](https://huggingface.co/datasets/somundane/egoverse-cup50)), every signal on every clip

| signal | tagged success | failure prevalence |
|---|---|---|
| SAM 3 cup-in-saucer | 36/50 | 28% |
| Qwen3-VL-8B | 39/50 | 22% |
| PaliGemma2-3B | 42/50 | 16% |
| Cosmos-Reason2-8B | 34/50 | 32% |
| **Fused** = mean(SEG, mean of VLMs) | **35/50** | **30%** |

Agreement: **27/50 clips unanimous** across all four signals, 37/50 with ≥3 of 4, only 6 split 2–2.
Per-episode votes: [`results/cup50_final.md`](results/cup50_final.md).

**One 2-minute aria egocentric video, split into attempts** — cup rests *relative to the saucer*
(so head motion cancels), each pick-up→set-down scored: **25 attempts → 16 success / 9 failure**,
plus 16 returns to the start position correctly not counted.
[`results/aria_split_2025-11-24-23-59-28-546000.md`](results/aria_split_2025-11-24-23-59-28-546000.md)

**Three things we learned (and can prove)**

1. **Ask about the goal *state*, not "was the task completed?"** The latter is unanswerable from
   one frame; p(yes) sat at 10⁻⁴ on every frame while the *ranking* still worked (AUROC 0.73 →
   0.83 after the fix). And the smaller the model, the more literal the question must be:
   PaliGemma hedged at 0.28–0.65 on the generic prompt and became decisive (0.07–0.97) on
   "is the cup resting on the saucer?". Only visible because we read logits — 0.49 and 0.02
   both print "No".
2. **Robot failures here are not drops.** Zero gripper slips across all 10 failures; hesitation
   (off-hand speed, dominant-arm path length) separates the classes. Geometry flags, VLM adjudicates.
3. **The unit of evaluation must match the video.** A cup50 clip is one attempt — score its end.
   An aria episode is 25 attempts — segment first, then score each; the episode-level number was
   never the right question. Camera motion has to cancel to segment: cup speed relative to a
   static scene object, not in image space.

**Cost:** geometric track is CPU, seconds. VLM at 1 fps is one batched forward pass per frame
(16/pass) on an H100; all 50 clips × 3 models ran in minutes. A full-dataset sweep is a `.map()`.

Graphs, ROC, per-episode verdicts and each model's actual top tokens: [`results/`](results/README.md).
Sample overlays and confidence-meter videos: [`demo/`](demo/).

## What we built

Three signals, one fusion step, every stage a Modal app under `multi-model-modal/`:

| stage | folder | model | what it produces |
|---|---|---|---|
| Segmentation | `sam3/` | **SAM 3** (text-prompted) | per-frame masks + stable track ids for `hand`, `cup`, `saucer` |
| VLM critic | `vlm_critic/` | **Qwen3-VL-8B** · **Cosmos-Reason2-8B** · **PaliGemma 2** · **Cosmos3-Nano** (vLLM) | per-frame `p(goal state reached)` read from **logits**, not text — the confidence meter |
| Geometric | `geometric/` | none — proprioception | holds / releases / slips / handover from `obs_gripper`, `cmd_gripper`, EE pose (robot episodes) |
| Fusion + eval | `fuse/` | numpy | tag per episode, accuracy + AUROC where labels exist, prevalence where they don't |

Every trace carries `source_indices` (sampled position → original frame) so any event can be
located in the source video, and overlay / confidence-meter videos are rendered **inside the
container** and written to both a local `<model>_out/` folder and the `egoverse-outputs` Volume.

## Results

**Labelled dev slice** — 20 `eva_bimanual` cup-on-saucer episodes, 10 success / 10 failure:

| criterion | accuracy | AUROC |
|---|---|---|
| VLM `p_done_late` (Qwen3-VL-8B, goal-state question, median-calibrated) | 0.70 | **0.83** |
| SEG cup-centroid-in-saucer-box over the last quarter | 0.70 | 0.64 |
| Fused (mean of VLM+SEG, geometric veto) | 0.70 | 0.74 |

**Unlabelled prevalence audit** — 50 human cup-on-saucer episodes
([somundane/egoverse-cup50](https://huggingface.co/datasets/somundane/egoverse-cup50)):
SEG alone **36/14 (28% failure)**; SAM 3 + all three VLMs fused **35/15 (30% failure)**, 27/50 unanimous.

**Human sets, three critics** (8 episodes each, `p(done)` ≥ 0.5 → success):

| set | statistic | Qwen3-VL-8B | PaliGemma2-3B | Cosmos-Reason2-8B |
|---|---|---|---|---|
| cup50 clips (n=50) | `late` (end of clip) | 39/50 success | 42/50 | 34/50 |
| aria 80–140 s egocentric | `max` (peak) | 8/8 success | 8/8 | 6/8 |

Why `max` for aria: the person looks away after placing the cup, so the goal state is not in
view at the end and `late` reads No for everything (0/8). The same statistic on the labelled
eva slice over-calls success (17/20), because a cup passing over the saucer mid-attempt also
peaks — so it is a per-dataset choice, stated, and both counts are shown for every set.

**Final verdict on cup50 — SAM 3 + all three VLMs combined, all 50 clips.** Success = the cup
ends up on the saucer; failure = it does not. `final = mean(SEG, mean(VLMs)) ≥ 0.5`, geometry and
VLMs weighted equally so neither can outvote the other alone. **35 success / 15 failure → 30%
failure prevalence.** Agreement is high: 27/50 clips are unanimous across all four signals, 37/50
have ≥3 of 4 agreeing; only 6 are split 2–2. Table with per-episode votes:
[`results/cup50_final.md`](results/cup50_final.md).

**One aria video split into its attempts** — a 2-minute egocentric episode is not one demo,
it's the person placing the cup again and again. Segmenting on cup rests *relative to the
saucer* (so head motion cancels) and scoring each pick-up→set-down transition:
**25 attempts → 16 success / 9 failure**, plus 16 returns to the start position that are
correctly not counted. Table and trace plot: [`results/aria_split_2025-11-24-23-59-28-546000.md`](results/aria_split_2025-11-24-23-59-28-546000.md).

**Cross-model report with graphs: [`results/`](results/README.md)** — ROC curves, p(done) traces per episode, tags per dataset × model (Qwen3-VL-8B, PaliGemma 2, Cosmos-Reason2-8B), model agreement, and what each model's top tokens actually were.

Full tables: [`demo/eva_fusion_summary.md`](demo/eva_fusion_summary.md),
[`demo/cup50_prevalence_summary.md`](demo/cup50_prevalence_summary.md).

**Sample SAM 3 overlays** — masks tinted per track, boxes, `id:score`, prompted with
`hand, cup, saucer` (click through to play; GitHub's file viewer plays `.mp4` inline):

- [Robot (`eva_bimanual`), 6 fps](demo/sam3_eva/2026-03-04-19-11-58-058000_overlay.mp4)
- [Human (cup50 dev slice), 10 fps](demo/sam3_10fps/692e98927641010d04354574_overlay.mp4)
- [Human (head-mounted), 1 fps](demo/sam3_10fps/human_1fps_2025-12-25-20-00-08-755000_overlay.mp4)

Full set: [`demo/`](demo/).

### Findings worth defending

- **Ask the VLM about the goal *state*, not task completion.** "Has the task been completed?"
  is unanswerable from one frame (a handover is invisible) — p(yes) sat at ~1e-4 on every frame
  of every episode, while the *ranking* still carried signal (AUROC 0.73). Asking "is the cup
  resting on the saucer?" lifted AUROC to 0.83. Reading logits rather than generated text is
  what made this diagnosable: 0.49 and 0.02 both print "No".
- **On this robot task, failures are not drops.** Zero gripper slips in all 10 failures. What
  separates the classes is effort shape — higher off-hand speed, shorter dominant-arm path in
  successes. Geometry flags hesitation; the VLM adjudicates outcome.
- **Conventions were checked, not assumed.** Gripper polarity (0 = closed) verified on the
  wrist camera; each arm's EE pose is in its own base frame, resolved through `extrinsics⁻¹`
  and verified by projecting onto the front image; SAM 3's saucer box must be per-frame on a
  head-mounted camera (an episode-median box tagged 50/50 human episodes as failure).
- **Same critic, robot and human.** Nothing was retrained between the fixed-camera bimanual
  robot slice and the head-mounted human slices; only the prompts changed.

## Run it

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
