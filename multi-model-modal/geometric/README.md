# Geometric track (stage [A]) — proprioception, no model

Success/failure features derived from the robot's own gripper and end-effector logs.
**No vision model, no GPU.** The whole 20-episode slice runs in seconds for cents.

```bash
modal run geometric/app.py
modal run geometric/app.py --limit 2
```

Writes `geometric_out/<episode>.json`, then prints a feature-separation ranking against the
labels in `task_name`.

## Why this replaces the planned stage [A]

The README's stage [A] assumes human egocentric video: MANO keypoints for grasp aperture,
Dyn-HaMR to cancel head motion. The eva slice has neither — two robot arms, fixed camera, no
hands. But it ships something strictly better for the same purpose:

| planned (human) | available here (robot) |
|---|---|
| grasp aperture estimated from MANO thumb–index distance | `obs_gripper`, **measured**, [0,1] |
| hand velocity from Dyn-HaMR world-frame fit | `obs_ee_pose` xyz, metres |
| — | `cmd_gripper` / `cmd_ee_pose` — what the arm was *told* to do |

That last row has no human analogue and is the most useful thing in the dataset.

## Conventions — verified, not assumed

Every one of these was checked against the data before being relied on. Two of them were
the opposite of my first guess.

- **Gripper: 0 = closed, 1 = open.** Confirmed on the wrist-camera images (fingers touching
  at 0.000, at the frame edges at 1.0). Closed-on-air reads 0.000–0.006. Blocked on the cup
  body plateaus at 0.026–0.101 depending on grasp point; pinching the cup *rim* reads as low
  as 0.017.
- **Each arm's `obs_ee_pose` is in its own base frame.** Both arms' x-range starts at the
  identical 0.095, and `extrinsics` carries a separate 4×4 per arm. Raw `|L − R|` is
  meaningless. `extrinsics[arm]` is the camera pose in that arm's frame, so
  `p_cam = inv(extrinsics[arm]) @ p_arm` puts both in one frame — verified by projecting
  through `intrinsics` onto the front image (100% of frames in-image, left arm at u≈194,
  right at u≈406, same height; the other direction put the left arm off the top edge).
- **Grippers don't always start closed.** Earlier-date episodes start at 0.00; later ones
  start open (~0.97). Any per-episode "closed floor" estimate breaks on the latter, because
  the right arm is often only ever commanded closed *while holding*. Thresholds are fixed
  constants for that reason.
- **NaN is real.** Three episodes have all-NaN `left.cmd_gripper`; one also `left.obs_gripper`.
  Every reduction is nan-aware, cmd-dependent logic falls back to obs-only, and each output
  lists `nan_fields`.

## The hold / release / slip model

A parallel gripper commanded fully closed stops early when something is between the fingers.

- **holding** — commanded closed, `obs > HOLD_EPS` (0.01), fingers *settled*
  (`|obs[t] − obs[t−3]| < 0.01`), for ≥ 0.3 s. The settle + duration requirements matter:
  closing on nothing sweeps `obs` 0.25→0 in ~6 frames and would otherwise register as a
  6-frame "hold" ending in a "slip".
- **release** — the hold ends and `cmd` opens within 0.5 s. Intentional.
- **slip** — the hold ends, `cmd` stays closed, and `obs` reaches the air band (≤ 0.006).
  The object left on its own. This is the robot-side drop signature.
- **ambiguous** — the hold ends into (0.006, 0.01] with `cmd` closed. Reported as its own
  count, not forced into either bucket.

`hold_width_median` (the plateau value) is a free proxy for *where* the cup was grasped.

## Handover

*"pick up cup with one arm, hand it over to the other arm, place it on the saucer."*
`handover_detected` = both arms holding simultaneously with camera-frame EE distance
< 0.30 m. Kept as an explicit boolean so it stays auditable.

**It is a feature, not a rule.** Several labelled *successes* complete with one arm and no
handover at all (e.g. `2026-03-04-19-11-58-058000`: right arm holds for 16.7 s and places;
the left arm's only closes are two missed grasps). Do not gate success on it.

## What the slice actually shows

Run locally over all 20 episodes (real function bodies via `.local()`, not a reimplementation):

- **Zero slips anywhere — including all 10 failures.** cup-on-saucer failures here are not
  drops. The "failure is legible in the hand" claim, in its slip form, does not hold on this
  task; failures are task-level (wrong placement, never grasped, gave up).
- Every success has ≥ 1 hold; two failures have none (`any_hold`: 1.0 vs 0.8).
- What separates the classes is *effort shape*: successes have higher left-arm speed
  (`left.ee_speed_mean` +0.96 SD), more left holds (+0.80), and a **shorter** right-arm path
  (−0.76 SD: 1.00 m vs 1.41 m). Failures look like one-armed struggling and re-reaching, not
  like dropping.

That is the honest input to the fusion stage: geometry flags *hesitation*, and the VLM
critic has to adjudicate outcome.

## Output

Per arm (`left`, `right`): `closed_reading_p5` (diagnostic), `closed_fraction`,
`hold_fraction`, `n_holds`, `longest_hold_s`, `hold_width_median`, `n_slips`, `n_releases`,
`n_ambiguous_ends`, `first_slip_frame`, `cmd_gripper_available`, `ee_speed_mean/max`,
`ee_path_length_m`, `track_err_mean/max`.

Per episode (`bimanual`): `both_holding_fraction`, `longest_handover_s`,
`n_handover_candidates`, `handover_frame`, `handover_detected`, `min_hand_distance_m`
(camera frame), `hand_distance_at_handover_m`, `any_hold`, `total_slips`,
`last_hold_end_frame` (aligns the VLM trace to the physical end of the attempt).

The separation report is in pooled SDs, not p-values — n=20 doesn't support significance
claims. Use it to decide which features to hand the fusion stage before training anything.
