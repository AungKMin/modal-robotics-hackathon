# Prevalence audit — 50 episodes tagged: 36 success / 14 failure (**28% failure prevalence**)

_No ground-truth labels in this set; accuracy/AUROC not computed._

| criterion | n | accuracy | AUROC | note |
|---|---|---|---|---|
| VLM  p_done_late > median=0.5000 | 0 | — | — | no outputs found |
| SEG  cup-on-saucer >= 0.50 | 0 | — | — | no outputs found |
| FUSED mean(VLM,SEG) w/ GEO veto | 0 | — | — | no outputs found |

## Per-episode

| episode | label | p_done_late | seg_score | any_hold | fused | pred | ok |
|---|---|---|---|---|---|---|---|
| 2025-11-14-16-19-50-305000 | None | — | 0.50 | — | 0.50 | success |  |
| 2025-11-14-16-39-40-009000 | None | — | 0.44 | — | 0.44 | failure |  |
| 2025-11-24-19-42-18-324000 | None | — | 0.28 | — | 0.28 | failure |  |
| 2025-11-30-16-25-06-265000 | None | — | 0.04 | — | 0.04 | failure |  |
| 2025-12-24-19-36-56-086000 | None | — | 0.30 | — | 0.30 | failure |  |
| 2025-12-25-20-00-08-755000 | None | — | 1.00 | — | 1.00 | success |  |
| 2025-12-26-00-56-41-044000 | None | — | 1.00 | — | 1.00 | success |  |
| 2025-12-26-18-05-07-214000 | None | — | 0.60 | — | 0.60 | success |  |
| 2025-12-26-18-31-01-838000 | None | — | 0.57 | — | 0.57 | success |  |
| 2026-01-11-18-13-58-430000 | None | — | 0.83 | — | 0.83 | success |  |
| 2026-01-11-23-11-22-998000 | None | — | 0.00 | — | 0.00 | failure |  |
| 2026-01-20-19-49-03-357000 | None | — | 0.27 | — | 0.27 | failure |  |
| 2026-01-24-05-29-04-636000 | None | — | 0.35 | — | 0.35 | failure |  |
| 692e98927641010d04354574 | None | — | 1.00 | — | 1.00 | success |  |
| 692e9b6668228e362d908f0e | None | — | 1.00 | — | 1.00 | success |  |
| 692e9fa092a31767e35da22c | None | — | 1.00 | — | 1.00 | success |  |
| 692ea164e2322e3b092b5dd8 | None | — | 1.00 | — | 1.00 | success |  |
| 692ea2012c8fefa9948e8dd0 | None | — | 1.00 | — | 1.00 | success |  |
| 692ea3beffdc0ca6345c4246 | None | — | 1.00 | — | 1.00 | success |  |
| 692ea4da74b24813e759755d | None | — | 1.00 | — | 1.00 | success |  |
| 692ea671dbc4294a49cc727e | None | — | 0.00 | — | 0.00 | failure |  |
| 692ea6ffc621d7f4aac3aafc | None | — | 0.00 | — | 0.00 | failure |  |
| 692ea773b77ab81b8b41ee87 | None | — | 1.00 | — | 1.00 | success |  |
| 692ea7b368228e362d90908e | None | — | 1.00 | — | 1.00 | success |  |
| 692ea7ff4e7eab2cafd26992 | None | — | 1.00 | — | 1.00 | success |  |
| 692ea886e2322e3b092b5ee5 | None | — | 1.00 | — | 1.00 | success |  |
| 692ea8fb727c13b350cb7cb8 | None | — | 1.00 | — | 1.00 | success |  |
| 692ea92a99488ff84776f987 | None | — | 0.00 | — | 0.00 | failure |  |
| 692ea97bea43f09edf26f5fb | None | — | 0.00 | — | 0.00 | failure |  |
| 692ea9bc95aad87d3e34466a | None | — | 1.00 | — | 1.00 | success |  |
| 692ea9ffaec602a46af10605 | None | — | 1.00 | — | 1.00 | success |  |
| 692eaa05c7cb0e94dc84bbf5 | None | — | 1.00 | — | 1.00 | success |  |
| 692eaa59a0e165ab2e42e516 | None | — | 0.00 | — | 0.00 | failure |  |
| 692eaa5fd3d807884d4dd7ae | None | — | 1.00 | — | 1.00 | success |  |
| 692eaa76dfa4113987776a57 | None | — | 0.00 | — | 0.00 | failure |  |
| 692eaac0c621d7f4aac3ab9d | None | — | 1.00 | — | 1.00 | success |  |
| 692eab1cb77ab81b8b41eeec | None | — | 1.00 | — | 1.00 | success |  |
| 692eac0739719ab57395b969 | None | — | 1.00 | — | 1.00 | success |  |
| 692eac2caec602a46af1065e | None | — | 1.00 | — | 1.00 | success |  |
| 692eac7274b24813e759760d | None | — | 1.00 | — | 1.00 | success |  |
| 692ead5d3019385fbbf5683e | None | — | 1.00 | — | 1.00 | success |  |
| 692ead5e3019385fbbf5684a | None | — | 1.00 | — | 1.00 | success |  |
| 692eae07424dbf22e75ed141 | None | — | 1.00 | — | 1.00 | success |  |
| 692eae54c621d7f4aac3abd6 | None | — | 1.00 | — | 1.00 | success |  |
| 692eae813019385fbbf56873 | None | — | 1.00 | — | 1.00 | success |  |
| 692eaeaa40338017b7f999ff | None | — | 1.00 | — | 1.00 | success |  |
| 692eaeba9cebf0ec017ddd36 | None | — | 1.00 | — | 1.00 | success |  |
| 692eaee539719ab57395b989 | None | — | 1.00 | — | 1.00 | success |  |
| 692eafd7eee2f8cb10d0f8e0 | None | — | 1.00 | — | 1.00 | success |  |
| 692eafd9eee2f8cb10d0f8f7 | None | — | 0.00 | — | 0.00 | failure |  |
