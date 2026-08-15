# VLM critic — p(task complete) from token probabilities

Deliverables #1 and #2 in one pass. Per sampled frame, ask "Has the task been completed?"
and read `p("Yes")` out of the **logits**, never the generated text. The curve over time is
the confidence meter; its late-frame value is the episode verdict.

## Run

```bash
modal run vlm_critic/app.py --limit 2 --max-frames 10    # shakeout
modal run vlm_critic/app.py --stride 30                  # full slice, Qwen3-VL
modal run vlm_critic/app.py --model cosmos --stride 30   # Cosmos Reason 2
```

Requires the `egoverse-episodes` Volume (see `../multi-model-modal/sam3/README.md`).

## Models

| key | id | notes |
|---|---|---|
| `qwen` (default) | `Qwen/Qwen3-VL-8B-Instruct` | apache-2.0, ungated |
| `cosmos` | `nvidia/Cosmos-Reason2-8B` | **gated** — accept terms first |

Cosmos Reason 2 is a fine-tune *of* Qwen3-VL-8B-Instruct — same `qwen3_vl` architecture, same
8.77B params — so one code path serves both and the comparison is genuinely apples-to-apples:
identical prompt, identical logit extraction, only the weights differ. That makes it a clean
ablation for "does physical-AI pretraining actually help judge manipulation?"

Both are Instruct, not Thinking, on purpose: chain-of-thought is pure cost when the only
thing read is a single distribution over the first answer token.

## Why logits, not generated text

A generated "No" gives you the argmax and nothing else. `p(yes)=0.49` and `p(yes)=0.02` both
print "No" but mean completely different things — and only the number is calibratable,
thresholdable, and plottable. This is the TOPReward trick, and it's why deliverable #2 comes
free rather than needing a trained head.

Two details that decide whether the trace is real signal or noise:

- **Token-id resolution.** `Yes` / ` Yes` / `yes` / `YES` are different ids, and which one a
  model emits varies by vocabulary. Betting on one id yields a trace of near-zero
  probabilities that looks like plausible data. `_variant_ids` collects every single-token
  surface form and sums their mass.
- **Renormalisation over {Yes, No}.** Raw `p(yes)` also reflects mass the model spends on
  unrelated continuations, which makes the trace sensitive to formatting quirks. Dividing by
  `p(yes) + p(no)` isolates the actual decision.

## Output

`vlm_critic_out/<episode>.json` per episode:

- `p_yes` — the trace, one value per sampled frame (**this is the confidence meter**)
- `source_indices` — maps trace position back to original episode frame number
- `label` — ground truth from `task_name`
- `p_yes_final`, `p_yes_max`, `p_yes_last_quartile_mean` — episode-level summaries

Summaries are late-weighted deliberately. A task is judged by how it *ended*; a mean over the
whole episode washes out exactly the evidence that matters, since every episode starts
incomplete and therefore starts near p(yes)=0.

The entrypoint prints a success-vs-failure separation check at the end. It is not the real
eval — that's AUROC plus a reliability curve — but it answers the only question worth asking
before spending more GPU: does the signal separate the classes at all? Overlapping means say
the prompt is wrong, and no fusion stage will rescue that.

## Cost

One full VLM forward pass per frame — much heavier per frame than SAM 3. `--stride 30` is one
frame per second at 30fps, giving ~700 forward passes across the 20-episode slice. Start
there; `--stride 15` doubles temporal resolution of the confidence meter and doubles cost.

GPU is `A100-40GB`: 8.77B params in bf16 is ~17.5GB of weights before activations and the
vision tower. L40S (48GB) also fits and is often easier to schedule.
