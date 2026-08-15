# Human Reward Model — EgoVerse Track 3

A success/failure critic for egocentric human demonstrations. Given a video segment from
[EgoVerse](https://github.com/GaTech-RL2/EgoVerse) and its annotations, we output a per-episode
success/failure/drop label, a dense confidence trace over time, and a prevalence audit of how much
of the dataset is actually failed demonstration.

Built for the EgoVerse Data Optimization & Evaluation Suite hackathon (one-day sprint), running on
[Modal](https://modal.com).

---

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
keypoint estimation. Runs on every frame to get hands and the manipulated object in 2D. This is our
throughput backbone: it's fast enough to sweep the whole dataset on Modal GPUs, which is what makes
the prevalence audit possible at all rather than a sampled estimate.

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

**NVIDIA Cosmos Reason** ([repo](https://github.com/nvidia-cosmos/cosmos-reason1),
[docs](https://docs.nvidia.com/cosmos/latest/introduction.html)) — an open reasoning VLM built
specifically for physical AI and embodied decisions, with long chain-of-thought over video. Cosmos
Reason 2 ships at 2B and 8B with long context. This is the single best open substitute for
Gemini Robotics-ER on this task: it's *designed* to judge whether a physical action makes sense,
which is exactly the critic role. Fits on one A100/H100 on Modal. **Strong recommend — make this
the open baseline and use Gemini as the oracle to measure the gap.**

**Qwen3-VL** — the strongest general open VLM for video Q&A at hackathon scale, and notably the
backbone that recent zero-shot progress-estimation work is built on. The relevant trick:
**TOPReward** ([paper](https://arxiv.org/html/2602.19313)) reads task progress directly out of a
VLM's *token probabilities* rather than its text output — asking "has the task been completed?" and
taking `p(yes)` as a dense reward. It reports 0.945 mean VOC on ManiRewardBench with Qwen3-VL and
beats GVL on open VLMs. This gives us a **free, training-free confidence meter**: run the prompt per
frame, plot `p(yes)` over time, and the curve *is* deliverable #2. Highest ratio of demo impact to
implementation time on this list.

**V-JEPA 2** ([repo](https://github.com/facebookresearch/vjepa2),
[HF](https://huggingface.co/docs/transformers/main/model_doc/vjepa2)) — Meta's self-supervised video
world model. Two uses: (a) frozen encoder features as input to a light success/failure probe, which
trains in minutes on our dev slice, and (b) *prediction error as a failure signal* — the model
predicts future latent states, and demos that go wrong are exactly the ones where the prediction
breaks down. Use (b) is the more interesting demo and needs no labels at all.

### Tier 2 — solid supporting pieces

**SAM 2** — video object segmentation with memory. Gives persistent object masks across the
episode, which upgrades our hand–object distance feature from keypoint-proximity to real
mask-level contact, and makes "the object left the frame / hit the floor" directly measurable.
Cheap to run, meaningfully improves the geometric track.

**HaMeR / WiLoR** — single-frame hand mesh recovery. Worth having as the initialization and
fallback for Dyn-HaMR, which is a multi-stage optimization pipeline and is the slowest, most
fragile thing in our stack. If Dyn-HaMR doesn't converge in time on hackathon day, a per-frame hand
mesh plus the EgoVerse-provided head pose recovers most of the signal.

**DINOv3 / SigLIP 2** — frozen image features for a nearest-neighbor "does this ending look like
other endings labeled success?" retrieval baseline. Two hours of work, surprisingly hard to beat,
and an excellent sanity check that the fancy stack is earning its cost.

**VideoMAE V2 / InternVideo2** — conventional video classification backbones. If we get enough
labels, fine-tuning one of these is the boring-but-strong baseline that a judge will ask about.
Have the number ready.

### Tier 3 — read the paper, probably don't build it today

**RARM** ([site](https://rarm-robotics.github.io/)) — turns a *single* successful demo into a dense
progress-aware reward via a contrastive temporal comparator, with no per-task engineering. Almost
perfectly shaped for EgoVerse (which has task groupings and therefore reference demos), but it's a
train-something project, not an inference project.

**Robo-Dopamine** ([paper](https://arxiv.org/html/2512.23703v1)) — general process reward modeling,
reports ~92.8% progress accuracy and 0.953 value-order consistency. Same note: great target, wrong
day.

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

## Repo layout

```
modal_app/
  image.py            # Modal image defs: rfdetr, dyn-hamr, vlm serving
  extract.py          # per-episode geometric feature extraction (fan-out)
  critic.py           # VLM critic endpoint (Cosmos Reason / Qwen3-VL served on Modal)
  fuse.py             # calibration + episode labeling
  audit.py            # dataset-level prevalence rollup
data/
  sync.py             # wraps EgoVerse egomimic/scripts/data_download/sync_s3.py
  schema.py           # zarr episode reader: images, MANO keypoints, poses, intrinsics
eval/
  dev_labels.jsonl    # ~100 hand-labeled episodes (our ground truth)
  metrics.py          # AUROC, reliability curve, per-slice prevalence
dashboard/
  app.py              # confidence meter + prevalence view
```

## Setup

```bash
uv venv --python 3.11 && uv pip install -e .
modal setup
aws configure                      # EgoVerse data lives in S3
python -m data.sync --tag aria-fold-clothes
```

Run the pipeline:

```bash
modal run modal_app/extract.py --tag aria-fold-clothes    # fan-out feature extraction
modal run modal_app/critic.py  --episodes out/shard-*.jsonl
modal run modal_app/audit.py                              # prevalence report
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
- V-JEPA 2 — https://github.com/facebookresearch/vjepa2
- TOPReward — https://arxiv.org/html/2602.19313
- RARM — https://rarm-robotics.github.io/
- Robo-Dopamine — https://arxiv.org/html/2512.23703v1
