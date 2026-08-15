# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state

This repo is **pre-implementation**. It contains a detailed design document (`README.md`), a pitch
deck PDF, and a single empty file (`multi-model-modal/modalscript.py`). There is no `pyproject.toml`,
no tests, and no Python package yet — the setup and run commands in `README.md` describe the
*intended* pipeline, not something that currently works.

Two consequences worth knowing before acting:

- The repo layout in `README.md` (`modal_app/`, `data/`, `eval/`, `dashboard/`) is a plan. Actual
  code so far lives under `multi-model-modal/`. When creating files, ask which layout to follow
  rather than silently picking one.
- `uv pip install -e .` from the README will fail until a `pyproject.toml` exists.

## What this project is

A success/failure critic for egocentric human demonstrations from the
[EgoVerse](https://github.com/GaTech-RL2/EgoVerse) dataset, built for the EgoVerse Data Optimization
& Evaluation Suite hackathon (Track 3, one-day sprint), running on [Modal](https://modal.com).

Three deliverables, all derived from video + existing annotations with no new human labeling:

1. **Episode tagger** — `success` / `failure` / `drop` label per episode, with a reason string
2. **Confidence meter** — per-frame success-probability trace, with the failure-onset frame marked
3. **Prevalence audit** — dataset-level rollup of what fraction of each task/embodiment/lab slice failed

## Architecture (as designed)

Three tracks feeding a fusion stage. Reading `README.md` in full is worth it before making design
decisions; the short version:

- **[A] Geometric track** — the cheap deterministic signal. RF-DETR-Keypoint per frame for 2D hand +
  object keypoints (the throughput backbone that makes a full-dataset sweep affordable), plus
  Dyn-HaMR for camera-motion-corrected 4D hand motion. Yields six features: grasp aperture,
  hand–object distance and derivative, world-frame object velocity, contact state, wrist jerk,
  head-pose entropy. Scored with an explicit rule set plus a small GBM so decisions stay auditable.
- **[B] Semantic track** — a VLM adjudicates task completion over sampled keyframes, conditioned on
  the episode's language annotation. Gemini Robotics-ER is the hosted oracle (API call from a Modal
  function, does not run on Modal); an open critic (Cosmos Reason 2 8B or Qwen3-VL 8B-Instruct)
  self-hosted on Modal is the production path. Deliverable #2 is meant to come from *token
  probabilities*, not text output: ask "has the task been completed?" per frame and plot `p(yes)`
  (the TOPReward trick). That's why the plan specifies Instruct over Thinking variants — the
  chain-of-thought is thrown away when only the logprob is wanted.
- **[C] Fusion** — logistic model over geometric event scores, VLM verdict logprob, and
  annotation-derived priors, calibrated on a ~100-episode hand-labeled dev slice.

Load-bearing design claims to preserve when changing things:

- **Failure is legible in the hand.** Geometry first, VLM only as adjudicator — not the reverse.
- **Camera motion must be factored out.** EgoVerse is head-mounted; naive per-frame tracking
  conflates head motion with hand motion, which destroys the drop signal. This is why Dyn-HaMR (or a
  per-frame hand mesh + EgoVerse-provided head pose fallback) is in the stack.
- **Segmentation must be text-prompted, not point-prompted.** SAM 3's Promptable Concept
  Segmentation takes the object noun parsed from the episode's language annotation, so no per-episode
  human seeding step is needed. SAM 2 would require clicking a point on frame one of every episode,
  which makes the prevalence audit impossible at scale.
- **RF-DETR-Keypoint's shipped checkpoint does not fit this task out of the box.** It's a preview
  release pretrained on COCO *person* keypoints — 17 body joints, wrists included, individual
  fingers not — and in egocentric video most of the body is out of frame. The plan is to fine-tune
  it on hand keypoints using EgoVerse's own MANO annotations as labels (~1 hour budgeted). Do not
  wire the pretrained head in as if it gave hand articulation.
- **Report AUROC and a reliability curve, not accuracy.** Class balance is skewed. The full eval set
  is AUROC, reliability curve, failure-onset localization (median frame error vs. human-marked
  onset), a geometric-only / VLM-only / fused ablation, and **cost per 1k episodes on Modal per
  configuration** — a critic too expensive to sweep the dataset doesn't solve the prevalence problem.
- **Vision-language-action policies are explicitly out of scope** (π0, OpenVLA, GR00T). They
  generate actions; this project judges them, and their value functions aren't exposed usefully as
  an offline critic.

`README.md` also carries a ranked model shortlist with explicit size recommendations (which variant
to run and why) — consult it rather than defaulting to the largest checkpoint of anything.

## Commands

Modal is the execution substrate. Once the pipeline exists, the intended flow from `README.md`:

```bash
uv venv --python 3.11 && uv pip install -e .
modal setup
aws configure                      # EgoVerse data lives in S3
python -m data.sync --tag aria-fold-clothes

modal run modal_app/extract.py --tag aria-fold-clothes    # fan-out feature extraction
modal run modal_app/critic.py  --episodes out/shard-*.jsonl
modal run modal_app/audit.py                              # prevalence report
```

## Modal conventions

The design leans on four Modal properties; keep new code aligned with them:

- **Fan-out is the product.** The prevalence audit is `.map()` over a Modal function with a cheap GPU
  (A10G/L4) per shard. Anything that forces sequential processing over episodes undermines the
  central deliverable.
- **Per-stage hardware.** Each stage declares its own GPU in the same file — RF-DETR wants many cheap
  GPUs, Dyn-HaMR and the VLM critic want a few big ones.
- **Pin dependencies in the image.** Dyn-HaMR's SLAM + MANO + hand-prior stack belongs in a Modal
  image definition, not in local setup instructions.
- **Volumes cache zarr.** EgoVerse episodes sync from S3 once into a Modal Volume, warm thereafter.

## Data gotchas

- EgoVerse episodes are zarr: images, MANO keypoints, EE/wrist/head pose, intrinsics, language
  annotation. The MANO keypoints are supervision the dataset ships for free — they're the labels for
  the RF-DETR fine-tune, not just a runtime feature source.
- `data/sync.py` is planned as a thin wrapper around EgoVerse's own
  `egomimic/scripts/data_download/sync_s3.py`, not an independent S3 client.
- Camera intrinsics have been mandatory in every episode's `zarr.json` since 07/08/2026. Stale local
  caches hard-crash — re-sync rather than debug.
- "Failure" is under-defined upstream. This project uses a three-way
  `success / task-failure / object-drop` taxonomy and publishes the rubric with the numbers.

## Permissions

`.claude/settings.json` pre-allows the pipeline toolchain (`modal`, `uv`, `python`, `pytest`,
`aws s3` read/copy/sync, `hf`) so it runs without prompting, and explicitly denies the destructive
counterparts — `aws s3 rm`/`rb`, `modal volume delete`, `modal app stop`, `git push --force`,
`git reset --hard`, `rm -rf`. Read the file for the exact list. Don't route around the deny rules.
