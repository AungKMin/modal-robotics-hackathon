# Prevalence audit — 20 episodes tagged: 12 success / 8 failure (**40% failure prevalence**)

# Fusion eval — 20 labelled episodes (10 success / 10 failure), VLM=qwen

| criterion | n | accuracy | AUROC | note |
|---|---|---|---|---|
| VLM  p_done_late > median=0.4171 | 20 | 0.70 | 0.83 | logit-derived, late window |
| SEG  cup-on-saucer >= 0.50 | 20 | 0.70 | 0.64 | fallback=cup_settled where no saucer track |
| FUSED mean(VLM,SEG) w/ GEO veto | 20 | 0.70 | 0.74 | any_hold=False -> failure |

GEO veto fired on 2 episode(s): 2026-03-01-21-53-09-065000(failure), 2026-03-04-19-45-48-423000(failure)
SEG mode(s) in use: ['cup_settled(fallback)']  ← re-run sam3 with --prompts "cup,saucer" for the geometric criterion

## Per-episode

| episode | label | p_done_late | seg_score | any_hold | fused | pred | ok |
|---|---|---|---|---|---|---|---|
| 2026-03-01-21-39-55-154000 | failure | 0.06 | 1.00 | True | 0.52 | success | ✗ |
| 2026-03-01-21-53-09-065000 | failure | 0.00 | 0.00 | False | 0.00 | failure | ✓ |
| 2026-03-02-15-45-51-437000 | failure | 0.24 | 0.22 | True | 0.18 | failure | ✓ |
| 2026-03-02-15-46-41-397000 | failure | 0.16 | 0.06 | True | 0.08 | failure | ✓ |
| 2026-03-02-16-08-11-055000 | success | 0.94 | 0.05 | True | 0.52 | success | ✓ |
| 2026-03-03-23-34-24-249000 | failure | 0.53 | 0.42 | True | 0.62 | success | ✗ |
| 2026-03-03-23-40-31-935000 | success | 0.87 | 0.60 | True | 0.80 | success | ✓ |
| 2026-03-03-23-41-02-407000 | success | 0.72 | 0.25 | True | 0.59 | success | ✓ |
| 2026-03-04-19-11-58-058000 | success | 0.31 | 0.06 | True | 0.12 | failure | ✗ |
| 2026-03-04-19-14-13-994000 | success | 0.56 | 0.90 | True | 0.87 | success | ✓ |
| 2026-03-04-19-45-48-423000 | failure | 0.00 | 0.06 | False | 0.00 | failure | ✓ |
| 2026-03-04-19-47-54-081000 | success | 0.73 | 0.82 | True | 0.88 | success | ✓ |
| 2026-03-04-20-25-12-236000 | failure | 0.21 | 0.15 | True | 0.14 | failure | ✓ |
| 2026-03-04-20-35-14-105000 | success | 0.98 | 1.00 | True | 1.00 | success | ✓ |
| 2026-03-04-21-20-46-356000 | failure | 0.11 | 0.36 | True | 0.21 | failure | ✓ |
| 2026-03-04-21-54-13-056000 | success | 0.11 | 1.00 | True | 0.53 | success | ✓ |
| 2026-03-04-22-00-39-903000 | success | 0.30 | 0.68 | True | 0.43 | failure | ✗ |
| 2026-03-04-22-36-13-297000 | failure | 0.57 | 1.00 | True | 0.92 | success | ✗ |
| 2026-03-04-23-11-19-672000 | failure | 0.84 | 0.96 | True | 0.98 | success | ✗ |
| 2026-03-04-23-18-15-776000 | success | 0.79 | 1.00 | True | 0.99 | success | ✓ |
