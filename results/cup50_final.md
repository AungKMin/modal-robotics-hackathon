# cup50 — final verdict (SAM 3 + VLMs)

**Success = the cup ends up on the saucer; failure = it does not.** 50 episodes · 9 with VLM scores (cosmos, paligemma, qwen) · rest SEG-only. final = mean(SEG, mean(VLMs)) ≥ 0.5. Statistic: `late`.

**Final: 36 success / 14 failure → 28% failure prevalence**

On the 9 episodes with all signals: 8 success / 1 failure.

| episode | label | SEG | cosmos | paligemma | qwen | final | votes ✅ | verdict |
|---|---|---|---|---|---|---|---|---|
| 2025-11-14-16-19-50-305000 | — | 0.50 | — | — | 0.53 | 0.51 | 2/2 | ✅ success |
| 2025-11-14-16-39-40-009000 | — | 0.44 | — | — | — | 0.44 | 0/1 | ❌ failure (SEG only) |
| 2025-11-24-19-42-18-324000 | — | 0.28 | — | — | — | 0.28 | 0/1 | ❌ failure (SEG only) |
| 2025-11-30-16-25-06-265000 | — | 0.04 | — | — | — | 0.04 | 0/1 | ❌ failure (SEG only) |
| 2025-12-24-19-36-56-086000 | — | 0.30 | — | — | — | 0.30 | 0/1 | ❌ failure (SEG only) |
| 2025-12-25-20-00-08-755000 | — | 1.00 | — | — | — | 1.00 | 1/1 | ✅ success (SEG only) |
| 2025-12-26-00-56-41-044000 | — | 1.00 | — | — | — | 1.00 | 1/1 | ✅ success (SEG only) |
| 2025-12-26-18-05-07-214000 | — | 0.60 | — | — | — | 0.60 | 1/1 | ✅ success (SEG only) |
| 2025-12-26-18-31-01-838000 | — | 0.57 | — | — | — | 0.57 | 1/1 | ✅ success (SEG only) |
| 2026-01-11-18-13-58-430000 | — | 0.83 | — | — | — | 0.83 | 1/1 | ✅ success (SEG only) |
| 2026-01-11-23-11-22-998000 | — | 0.00 | — | — | — | 0.00 | 0/1 | ❌ failure (SEG only) |
| 2026-01-20-19-49-03-357000 | — | 0.27 | — | — | — | 0.27 | 0/1 | ❌ failure (SEG only) |
| 2026-01-24-05-29-04-636000 | — | 0.35 | — | — | — | 0.35 | 0/1 | ❌ failure (SEG only) |
| 692e98927641010d04354574 | — | 1.00 | 0.96 | 0.58 | 1.00 | 0.92 | 4/4 | ✅ success |
| 692e9b6668228e362d908f0e | — | 1.00 | 0.94 | 0.32 | 1.00 | 0.88 | 3/4 | ✅ success |
| 692e9fa092a31767e35da22c | — | 1.00 | 0.75 | 0.41 | 0.92 | 0.85 | 3/4 | ✅ success |
| 692ea164e2322e3b092b5dd8 | — | 1.00 | 0.88 | 0.47 | 1.00 | 0.89 | 3/4 | ✅ success |
| 692ea2012c8fefa9948e8dd0 | — | 1.00 | 0.95 | 0.55 | 1.00 | 0.92 | 4/4 | ✅ success |
| 692ea3beffdc0ca6345c4246 | — | 1.00 | 0.32 | 0.47 | 0.00 | 0.63 | 1/4 | ✅ success |
| 692ea4da74b24813e759755d | — | 1.00 | 0.86 | 0.59 | 1.00 | 0.91 | 4/4 | ✅ success |
| 692ea671dbc4294a49cc727e | — | 0.00 | 0.73 | 0.44 | 0.56 | 0.29 | 2/4 | ❌ failure |
| 692ea6ffc621d7f4aac3aafc | — | 0.00 | — | — | — | 0.00 | 0/1 | ❌ failure (SEG only) |
| 692ea773b77ab81b8b41ee87 | — | 1.00 | — | — | — | 1.00 | 1/1 | ✅ success (SEG only) |
| 692ea7b368228e362d90908e | — | 1.00 | — | — | — | 1.00 | 1/1 | ✅ success (SEG only) |
| 692ea7ff4e7eab2cafd26992 | — | 1.00 | — | — | — | 1.00 | 1/1 | ✅ success (SEG only) |
| 692ea886e2322e3b092b5ee5 | — | 1.00 | — | — | — | 1.00 | 1/1 | ✅ success (SEG only) |
| 692ea8fb727c13b350cb7cb8 | — | 1.00 | — | — | — | 1.00 | 1/1 | ✅ success (SEG only) |
| 692ea92a99488ff84776f987 | — | 0.00 | — | — | — | 0.00 | 0/1 | ❌ failure (SEG only) |
| 692ea97bea43f09edf26f5fb | — | 0.00 | — | — | — | 0.00 | 0/1 | ❌ failure (SEG only) |
| 692ea9bc95aad87d3e34466a | — | 1.00 | — | — | — | 1.00 | 1/1 | ✅ success (SEG only) |
| 692ea9ffaec602a46af10605 | — | 1.00 | — | — | — | 1.00 | 1/1 | ✅ success (SEG only) |
| 692eaa05c7cb0e94dc84bbf5 | — | 1.00 | — | — | — | 1.00 | 1/1 | ✅ success (SEG only) |
| 692eaa59a0e165ab2e42e516 | — | 0.00 | — | — | — | 0.00 | 0/1 | ❌ failure (SEG only) |
| 692eaa5fd3d807884d4dd7ae | — | 1.00 | — | — | — | 1.00 | 1/1 | ✅ success (SEG only) |
| 692eaa76dfa4113987776a57 | — | 0.00 | — | — | — | 0.00 | 0/1 | ❌ failure (SEG only) |
| 692eaac0c621d7f4aac3ab9d | — | 1.00 | — | — | — | 1.00 | 1/1 | ✅ success (SEG only) |
| 692eab1cb77ab81b8b41eeec | — | 1.00 | — | — | — | 1.00 | 1/1 | ✅ success (SEG only) |
| 692eac0739719ab57395b969 | — | 1.00 | — | — | — | 1.00 | 1/1 | ✅ success (SEG only) |
| 692eac2caec602a46af1065e | — | 1.00 | — | — | — | 1.00 | 1/1 | ✅ success (SEG only) |
| 692eac7274b24813e759760d | — | 1.00 | — | — | — | 1.00 | 1/1 | ✅ success (SEG only) |
| 692ead5d3019385fbbf5683e | — | 1.00 | — | — | — | 1.00 | 1/1 | ✅ success (SEG only) |
| 692ead5e3019385fbbf5684a | — | 1.00 | — | — | — | 1.00 | 1/1 | ✅ success (SEG only) |
| 692eae07424dbf22e75ed141 | — | 1.00 | — | — | — | 1.00 | 1/1 | ✅ success (SEG only) |
| 692eae54c621d7f4aac3abd6 | — | 1.00 | — | — | — | 1.00 | 1/1 | ✅ success (SEG only) |
| 692eae813019385fbbf56873 | — | 1.00 | — | — | — | 1.00 | 1/1 | ✅ success (SEG only) |
| 692eaeaa40338017b7f999ff | — | 1.00 | — | — | — | 1.00 | 1/1 | ✅ success (SEG only) |
| 692eaeba9cebf0ec017ddd36 | — | 1.00 | — | — | — | 1.00 | 1/1 | ✅ success (SEG only) |
| 692eaee539719ab57395b989 | — | 1.00 | — | — | — | 1.00 | 1/1 | ✅ success (SEG only) |
| 692eafd7eee2f8cb10d0f8e0 | — | 1.00 | — | — | — | 1.00 | 1/1 | ✅ success (SEG only) |
| 692eafd9eee2f8cb10d0f8f7 | — | 0.00 | — | — | — | 0.00 | 0/1 | ❌ failure (SEG only) |
