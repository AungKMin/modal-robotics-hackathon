# Human Reward Model — EgoVerse Track 3

A success/failure critic for manipulation demos, running on [Modal](https://modal.com). Given
an EgoVerse episode (robot **or** human egocentric) it outputs a per-episode
success/failure tag, a per-frame confidence trace, and a dataset-level prevalence audit —
with no new human labelling and no model training.

Built in one day for the EgoVerse Data Optimization & Evaluation Suite hackathon.

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
([somundane/egoverse-cup50](https://huggingface.co/datasets/somundane/egoverse-cup50)), SEG criterion:
**36 success / 14 failure → 28% failure prevalence.**

Full tables: [`demo/eva_fusion_summary.md`](demo/eva_fusion_summary.md),
[`demo/cup50_prevalence_summary.md`](demo/cup50_prevalence_summary.md).
Sample SAM 3 overlays (10 fps human clips, 6 fps robot episodes): [`demo/`](demo/).

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

---

# Design notes (original plan)

The rest of this document is the design written before the sprint. Where the plan and the
built system diverge, the sections above and the per-folder READMEs are authoritative — in
particular the dev slice turned out to be a bimanual *robot* embodiment, so stage [A] was
rebuilt on gripper proprioception rather than MANO/Dyn-HaMR (see `geometric/README.md`).

## The problem

EgoVerse is human egocentric video collected at scale for robot learning. Scale creates a new
problem: **success and failure demos are mixed together in human data**. A pick that slips out of
the hand, a fold that never closes, a grasp that misses — all of it lands in the same bucket as
clean demos, and imitation learning happily fits the failures too.

We can't yet answer, deterministically:

- Did the human actually succeed?
- Where in the episode did it go wrong?
- How much of this dataset is quietly poisoned?

## What we build

Three deliverables, all from video + existing annotations, no new human labeling:

| # | Deliverable | Output |
|---|---|---|
| 1 | **Episode tagger** | `success` / `failure` / `drop` label per episode, with reason string |
| 2 | **Confidence meter** | Per-frame success-probability trace over a video segment, with the failure onset frame marked |
| 3 | **Prevalence audit** | Dataset-level report: what fraction of each task/embodiment/lab slice is failed, with per-slice breakdown |

The core claim we want to defend at demo time: **failure is legible in the hand.** A drop, a slip,
or a missed grasp shows up as a signature in hand pose, contact state, and object–hand relative
motion long before it shows up in a language annotation. So we build a physical, cheap signal
first, and use a VLM only as an adjudicator.

---

## Architecture

```
EgoVerse zarr episode (images + MANO keypoints + EE/wrist/head pose + intrinsics + language)
          │
          ├─► [A] Geometric track  ── RF-DETR-Keypoint ──► per-frame 2D hand + object keypoints
          │                        └─ Dyn-HaMR ──────────► 4D global hand motion (camera-motion-corrected)
          │                                                 ↓
          │                                          contact / release / velocity /
          │                                          grasp-aperture features
          │
          ├─► [B] Semantic track ── VLM critic over sampled frames ──► success? failure mode? onset?
          │
          └─► [C] Fusion ──► calibrated per-frame p(success) ──► episode label + confidence trace
                              │
                              └─► prevalence audit rolled up over the dataset
```

### [A] Geometric track — the cheap, deterministic signal

**RF-DETR-Keypoint** ([Roboflow](https://github.com/roboflow/rf-detr)) — real-time detection +
keypoint estimation, Nano (30.5M) through 2XLarge; Large is 129M. Runs on every frame to get hands
and the manipulated object in 2D. This is our throughput backbone: it's fast enough to sweep the
whole dataset on Modal GPUs, which is what makes the prevalence audit possible at all rather than a
sampled estimate. We plan to run **Small or Medium** — Large's accuracy isn't worth the latency when
the geometric track is a feature extractor, not the final judge.

> **Catch worth planning around:** RF-DETR-Keypoint currently ships as a preview with a single
> COCO-person-keypoints-pretrained checkpoint. COCO person keypoints are 17 *body* joints — wrists
> included, individual fingers not. That's not hand articulation, and in egocentric video most of
> the body is out of frame anyway, so the pretrained head is a poor fit as-is. The fix is the thing
> RF-DETR is built for: **fine-tune it on hand keypoints, using EgoVerse's own MANO annotations as
> labels.** The dataset ships the supervision we need for free. Budget ~1 hour of the sprint for
> this; it's the difference between RF-DETR being our backbone and being dead weight.

**Dyn-HaMR** ([CVPR 2025 Highlight](https://github.com/ZhengdiYu/Dyn-HaMR)) — recovers 4D global
interacting-hand motion from a *dynamic* camera. This matters more than it looks: EgoVerse is
head-mounted egocentric video, so the camera is always moving. Naive per-frame hand tracking
conflates head motion with hand motion, and "the object moved away from the hand" becomes
indistinguishable from "the head turned." Dyn-HaMR's SLAM stage factors out camera motion, so
hand-in-world trajectories are actually comparable across frames. That's what makes a drop
detectable: **object accelerates downward in world frame while hand aperture opens and hand
velocity does not match.**

Derived features per frame:

- grasp aperture (thumb–index distance, MANO-derived)
- hand–object distance and its derivative
- world-frame object velocity, particularly the gravity-aligned component
- contact state (heuristic: aperture below threshold ∧ hand–object distance below threshold)
- wrist jerk / tremor (correlates with fumbles)
- head-pose entropy (searching behavior after a drop — humans look for what they dropped)

A drop is a specific, checkable pattern in these six signals. We score it with an explicit rule set
plus a small gradient-boosted classifier over the feature window, so every decision is auditable —
which matters for the "is the method defensible?" judging criterion.

### [B] Semantic track — VLM adjudication

Geometry tells you *something changed*. It doesn't tell you whether the task was completed. For
that we sample keyframes (uniformly + at geometric-event boundaries) and ask a VLM, conditioned on
the episode's language annotation, whether the stated task was accomplished.

Planned primary: **Gemini Robotics-ER** for its embodied/spatial reasoning and pointing. It's an
API model, so it does not run on Modal — we use it as a hosted call from a Modal function and treat
it as the accuracy ceiling / labeling oracle rather than the production path.

Because a hosted model is a dependency risk in a one-day sprint (and a cost risk over a full-dataset
sweep), we also stand up an open, self-hosted critic on Modal. See **Model shortlist** below.

### [C] Fusion and calibration

Geometry and semantics disagree in informative ways. We fuse with a simple logistic model over
[geometric event scores, VLM verdict logprob, annotation-derived priors] and calibrate on a
hand-labeled dev slice (~100 episodes, labeled by the team in the morning) so the confidence meter
means something. Report reliability curve + AUROC, not just accuracy — the class balance is
skewed and accuracy would flatter us.

---

## Model shortlist — what else is worth running on Modal

Beyond RF-DETR-Keypoint, Dyn-HaMR, and Gemini Robotics-ER, these are the open models that fit this
track and self-host cleanly on Modal GPUs. Ranked by what we'd actually reach for.

### Tier 1 — highest leverage for Track 3

**NVIDIA Cosmos Reason 2** — **2B / 8B / 32B** ([repo](https://github.com/nvidia-cosmos/cosmos-reason1),
[2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B), [8B](https://huggingface.co/nvidia/Cosmos-Reason2-8B),
[32B](https://huggingface.co/nvidia/Cosmos-Reason2-32B)) — an open reasoning VLM built specifically
for physical AI and embodied decisions, with long chain-of-thought over video. Built on the
Qwen3-VL-Instruct architecture, so the two entries on this list share a serving path. This is the
single best open substitute for Gemini Robotics-ER on this task: it's *designed* to judge whether a
physical action makes sense, which is exactly the critic role. **Run the 8B** — fits comfortably on
one A100-40GB or H100 on Modal, and the 32B's extra quality isn't worth doubling GPU cost on a
one-day budget. The 2B is the fallback if we need to sweep the whole dataset rather than a slice.
**Strong recommend — make this the open baseline and use Gemini as the oracle to measure the gap.**

**Qwen3-VL** — **dense 2B / 4B / 8B / 32B**, plus MoE 30B-A3B (3B active) and 235B-A22B (22B active);
each size in Instruct and Thinking flavors. The strongest general open VLM for video Q&A at
hackathon scale, and the backbone recent zero-shot progress-estimation work is built on. The
relevant trick: **TOPReward** ([paper](https://arxiv.org/html/2602.19313)) reads task progress
directly out of a VLM's *token probabilities* rather than its text output — asking "has the task
been completed?" and taking `p(yes)` as a dense reward. It reports 0.945 mean VOC on ManiRewardBench
with Qwen3-VL and beats GVL on open VLMs. This gives us a **free, training-free confidence meter**:
run the prompt per frame, plot `p(yes)` over time, and the curve *is* deliverable #2. Highest ratio
of demo impact to implementation time on this list. **Use 8B-Instruct** — Thinking variants burn
tokens on chain-of-thought we throw away, since we only want the logprob.

**V-JEPA 2** ([repo](https://github.com/facebookresearch/vjepa2),
[HF](https://huggingface.co/docs/transformers/main/model_doc/vjepa2)) — Meta's self-supervised video
world model. Encoders: **ViT-L 300M / ViT-H 600M / ViT-g 1B**; the released encoder is the ViT-g, and
the full encoder+predictor world model is ~1.2B. V-JEPA **2.1** broadens this with distilled
ViT-B (80M) and a ViT-G (2B) at 384px. Two uses: (a) frozen encoder features as input to a light
success/failure probe, which trains in minutes on our dev slice, and (b) *prediction error as a
failure signal* — the model predicts future latent states, and demos that go wrong are exactly the
ones where the prediction breaks down. Use (b) is the more interesting demo and needs no labels at
all. **Start with ViT-L (300M)** for the probe; only reach for ViT-g if the probe underfits.

### Tier 2 — solid supporting pieces

**SAM 3** — **848M** ([Meta](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/),
[explainer](https://blog.roboflow.com/what-is-sam3/)) — video object segmentation, one Perception
Encoder + a DETR-based detector + SAM 2's dense-memory video tracker. Gives persistent object masks
across the episode, upgrading our hand–object distance feature from keypoint proximity to real
mask-level contact, and making "the object left the frame / hit the floor" directly measurable.
See the note below on why this replaces SAM 2 in the plan. SAM 3.1 is also out, with real-time
multiplexing — worth a look if throughput binds.

**HaMeR / WiLoR** — single-frame hand mesh recovery, both ViT-H-class backbones (~600M). Worth
having as the initialization and fallback for Dyn-HaMR, which is a multi-stage optimization pipeline
and the slowest, most fragile thing in our stack. If Dyn-HaMR doesn't converge in time on hackathon
day, a per-frame hand mesh plus the EgoVerse-provided head pose recovers most of the signal.

**DINOv3 / SigLIP 2** — frozen image features for a nearest-neighbor "does this ending look like
other endings labeled success?" retrieval baseline. Both ship from ViT-S/B (~22–86M) up through
billion-plus variants; **use a ViT-B or ViT-L** — this is a sanity baseline, not a place to spend
GPU. Two hours of work, surprisingly hard to beat, and an excellent check that the fancy stack is
earning its cost.

**VideoMAE V2 / InternVideo2** — conventional video classification backbones, ViT-B through ViT-g
(~1B) for VideoMAE V2, 1B and 6B for InternVideo2. If we get enough labels, fine-tuning a **ViT-B or
ViT-L** is the boring-but-strong baseline a judge will ask about. Have the number ready.

### Tier 3 — read the paper, probably don't build it today

**RARM** ([site](https://rarm-robotics.github.io/)) — turns a *single* successful demo into a dense
progress-aware reward via a contrastive temporal comparator, with no per-task engineering. Almost
perfectly shaped for EgoVerse (which has task groupings and therefore reference demos), but it's a
train-something project, not an inference project.

**Robo-Dopamine** ([paper](https://arxiv.org/html/2512.23703v1)) — general process reward modeling,
reports ~92.8% progress accuracy and 0.953 value-order consistency. Same note: great target, wrong
day.

### Why SAM 3 and not SAM 2

SAM 1 and SAM 2 return **one object per prompt**, and the prompt is geometric — a point, a box, a
mask. That means somebody has to say *where* the object is on frame one of every episode. Across
thousands of episodes, that's either manual clicking (impossible at our scale) or a hand-rolled
heuristic to seed the prompt (fragile, and one more thing to defend to a judge).

SAM 3 adds **Promptable Concept Segmentation**: give it a short noun phrase — "the towel", "the
mug" — and it returns masks and stable IDs for *every* matching instance at once, images and video
alike. It keeps SAM 2's memory-based tracker underneath, so we lose nothing on the tracking side.

That difference is decisive here specifically because **EgoVerse episodes carry language
annotations.** We can parse the object noun straight out of the task description and hand it to SAM
3 as the prompt. The segmentation stage becomes fully automatic and text-driven, with no seeding
step and no per-episode human input — which is the entire requirement for the prevalence audit.
Meta reports roughly 2× over prior systems on concept segmentation for image and video both.

Use SAM 2 only if SAM 3's licensing or weights availability turns out to be a blocker on the day.

### Explicitly not using

Vision-language-action policies (π0, OpenVLA, GR00T). They generate actions; we need to *judge*
them. Their value functions aren't exposed in a form that's useful as an offline critic here, and
loading one would eat our GPU budget for no deliverable.

---

## Why Modal

- **Fan-out is the whole product.** The prevalence audit means running the geometric track over
  thousands of episodes. `.map()` over a Modal function with an A10G/L4 per shard turns an
  overnight job into a coffee break.
- **Heterogeneous GPUs per stage.** RF-DETR wants many cheap GPUs; Dyn-HaMR and the VLM critic want
  a few big ones. Modal lets each stage declare its own hardware in the same file.
- **Container image = reproducibility.** Dyn-HaMR's dependency stack (SLAM + MANO + hand priors) is
  the kind of thing that eats an afternoon locally. Pin it once in an image, never think about it
  again.
- **Volumes for the zarr cache.** EgoVerse episodes get synced from S3 once into a Modal Volume and
  are then warm for every subsequent run.

---

## Repo layout (as built)

```
multi-model-modal/
  sam3/         app.py (clips) · episodes.py (zarr episodes, overlay video) · visualize.py
  vlm_critic/   app.py (Qwen3-VL / Cosmos-Reason2 / PaliGemma) · cosmos3.py (vLLM)
  geometric/    proprioception features, CPU
  fuse/         fuse.py — tags, accuracy/AUROC, prevalence
  cup50/        prepare.py — HF parquet → zarr Volume
demo/           sample overlays + result summaries
sync_s3.py      EgoVerse S3 sync with named episode presets
```

## Evaluation

We report on the held-out half of the hand-labeled dev slice:

- **AUROC** for success vs. failure (primary — class balance is skewed, accuracy would mislead)
- **Reliability curve** — the confidence meter has to be calibrated to be worth anything
- **Failure-onset localization** — median frame error vs. human-marked onset
- **Ablation** — geometric-only, VLM-only, fused. If the geometric track is doing all the work, we
  say so; if the VLM is, we say that too.
- **Cost per 1k episodes** on Modal, per configuration. A critic nobody can afford to run over the
  full dataset doesn't solve the prevalence problem.

## Known limitations

- Dev-set labels are ours, made in a few hours, on one task family. Prevalence numbers generalize
  as far as that slice does and no further.
- "Failure" is under-defined in EgoVerse. We use a three-way `success / task-failure / object-drop`
  taxonomy and publish the rubric alongside the numbers rather than pretending the boundary is crisp.
- Dyn-HaMR is an optimization pipeline, not a feedforward net. It is the throughput bottleneck and
  the most likely thing to be swapped for a per-frame fallback under time pressure.
- Camera intrinsics are mandatory in every EgoVerse episode's `zarr.json` as of 07/08/2026; stale
  local caches hard-crash. Re-sync rather than debug.

## Team

Four-person team, EgoVerse hackathon, Track 3.

## References

- EgoVerse dataset — https://github.com/GaTech-RL2/EgoVerse
- RF-DETR — https://github.com/roboflow/rf-detr
- Dyn-HaMR: Recovering 4D Interacting Hand Motion from a Dynamic Camera (CVPR 2025) — https://github.com/ZhengdiYu/Dyn-HaMR
- NVIDIA Cosmos Reason — https://github.com/nvidia-cosmos/cosmos-reason1
- SAM 3: Segment Anything with Concepts — https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/
- V-JEPA 2 — https://github.com/facebookresearch/vjepa2
- TOPReward — https://arxiv.org/html/2602.19313
- RARM — https://rarm-robotics.github.io/
- Robo-Dopamine — https://arxiv.org/html/2512.23703v1
