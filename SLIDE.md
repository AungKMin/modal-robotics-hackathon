# Human Reward Model — Track 3

## Results at a glance

**Does the demo succeed? Ask the VLM's logits, not its words.**
Per-frame `p(goal state reached)` read from Qwen3-VL / Cosmos-Reason2 / PaliGemma logits →
confidence meter → episode tag → prevalence audit. Plus SAM 3 text-prompted masks
(`hand`, `cup`, `saucer`) and gripper proprioception. Zero training, zero new labels, all on Modal.

**Labelled robot slice** — 20 eva cup-on-saucer episodes, 10 success / 10 failure

| signal | accuracy | AUROC |
|---|---|---|
| VLM p(done), last quarter (Qwen3-VL-8B) | 0.70 | **0.83** |
| SAM 3 cup-centroid-in-saucer | 0.70 | 0.64 |
| Fused + geometric veto | 0.70 | 0.74 |

**Unlabelled human video** — same critic, only the prompts changed

| | Qwen3-VL-8B | PaliGemma2-3B | Cosmos-Reason2-8B | SAM 3 |
|---|---|---|---|---|
| cup50 short clips (n=8, end-of-clip) | 8/9 success | 3/8 | 7/8 | **36/50 → 28% failure prevalence** |
| aria 2-min egocentric (n=8, peak) | 8/8 | 8/8 | 6/8 | 2/5 |

**Three things we learned**

1. **Ask about the goal *state*, not "was the task completed?"** The latter is unanswerable from
   one frame; p(yes) sat at 10⁻⁴ on every frame while the *ranking* still worked (AUROC 0.73 →
   0.83 after the fix). Only visible because we read logits — 0.49 and 0.02 both print "No".
2. **Robot failures here are not drops.** Zero gripper slips across all 10 failures; hesitation
   (off-hand speed, dominant-arm path length) separates the classes. Geometry flags, VLM
   adjudicates.
3. **The scoring window must match the video.** End-of-clip for short demos; peak for long
   egocentric footage where the person looks away after placing the cup. The same statistic on
   eva over-calls success (17/20), so it is chosen per dataset and both counts are shown.

**Cost:** geometric track is CPU, seconds. VLM at 1 fps is one forward pass per frame, batched
16, on L40S/A100. A full-dataset sweep is a `.map()`.

Graphs, ROC, per-episode verdicts and each model's actual top tokens: [`results/`](results/README.md).
Sample overlays and confidence-meter videos: [`demo/`](demo/).

