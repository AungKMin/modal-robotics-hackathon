# Human Reward Model — Track 3

## Results at a glance

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

