# VLM critic — p(task complete) from token probabilities

Deliverables #1 and #2 in one pass. Per sampled frame, ask "Has the task been completed?"
and read `p("Yes")` out of the **logits**, never the generated text. The curve over time is
the confidence meter; its late-frame value is the episode verdict.

## Run

```bash
modal run vlm_critic/app.py --limit 2 --max-frames 10    # shakeout
modal run vlm_critic/app.py --stride 30                  # full slice, Qwen3-VL
modal run vlm_critic/app.py --model cosmos --stride 30   # Cosmos Reason 2
modal run vlm_critic/cosmos3.py --stride 30              # Cosmos 3 reasoner via vLLM
```

Requires the `egoverse-episodes` Volume (see `../multi-model-modal/sam3/README.md`).

## Models

| key | id | serving | notes |
|---|---|---|---|
| `qwen` (default) | `Qwen/Qwen3-VL-8B-Instruct` | transformers, `app.py` | apache-2.0, ungated |
| `cosmos` | `nvidia/Cosmos-Reason2-8B` | transformers, `app.py` | **gated** — accept terms first |
| `cosmos3` | `nvidia/Cosmos3-Nano` (reasoner tower) | vLLM, `cosmos3.py` | 16B omnimodel, ungated (OpenMDW 1.1), H100 |

There is no "Cosmos Reason 3": the Reason line stops at Reason 2 (2B/8B/32B). Cosmos 3 is an
omnimodel whose autoregressive **reasoner** tower does understanding; NVIDIA serves it via
`vllm serve nvidia/Cosmos3-Nano --hf-overrides '{"architectures":["Cosmos3ReasonerForConditionalGeneration"]}'`.
`cosmos3.py` runs that in-container and reads p(Yes) from the OpenAI-compatible API's
`top_logprobs` on a single generated token — the same trick one layer up. It is a separate
app because vLLM pins its own torch and would fight the transformers image.

Why three: `qwen` vs `cosmos` isolates *fine-tuning* (same base). `cosmos3` isolates
*lineage* (different base entirely). If all three agree the signal is real.

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

`vlm_critic_out/<model_key>/<episode>.json` per episode, plus
`vlm_critic_out/<model_key>/<episode>_meter.mp4` — the **confidence meter as a video**: each
sampled frame with a p(done) bar, the trace-so-far sparkline, source frame and label. Rendered
in-container and also written to the `egoverse-outputs` Volume
(`modal volume ls egoverse-outputs /vlm_critic/<model_key>`). `--no-render` skips it.

Trace fields:

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
