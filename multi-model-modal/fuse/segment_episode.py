"""
Split one long episode into its attempts and score each one.

    uv run --with matplotlib --with numpy python fuse/segment_episode.py \\
        --episode 2025-11-24-23-59-28-546000 \\
        --seg-dir sam3_out/aria_split --vlm-dir vlm_critic_out_aria_split/qwen

A 2-minute aria episode is not one demo — the person places the cup on the saucer again and
again. Scoring the whole thing with one number hides that. This script:

  1. reads the SAM 3 trace and computes cup speed from the cup centroid;
  2. finds attempts = runs where the cup is moving (lifted, carried), merged across short
     gaps, each ending when the cup settles;
  3. scores each attempt in the window right after it settles:
       SEG  = fraction of window frames with the cup centroid inside the (per-frame) saucer box
       VLM  = mean p(done) over the window (matched by source frame index)
     tag = success if both available and mean(SEG, VLM) >= 0.5, else whichever exists.
  4. writes results/aria_split_<episode>.md and .png (traces with attempts shaded).

Motion-based segmentation is what makes failed attempts countable: a failed placement never
raises p(done), so segmenting on p(done) peaks would silently drop it.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from fuse import _tracks_for, CUP_WORDS, SAUCER_WORDS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "results"


def cup_centroid(frame):
    cups = [c for c in _tracks_for(frame, CUP_WORDS) if c.get("centroid_xy")]
    if not cups:
        return None
    return np.array(max(cups, key=lambda o: o["score"])["centroid_xy"], dtype=float)


def saucer_box(frame, pad=0.25):
    s = _tracks_for(frame, SAUCER_WORDS)
    if not s:
        return None
    x0, y0, x1, y1 = max(s, key=lambda o: o["score"])["box_xyxy"]
    pw, ph = pad * (x1 - x0), pad * (y1 - y0)
    return (x0 - pw, y0 - ph, x1 + pw, y1 + ph)


def runs(mask):
    if not mask.any():
        return []
    p = np.concatenate(([False], mask, [False]))
    e = np.flatnonzero(p[1:] != p[:-1])
    return list(zip(e[0::2].tolist(), e[1::2].tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--seg-dir", required=True)
    ap.add_argument("--vlm-dir", default=None)
    ap.add_argument("--speed-thresh", type=float, default=60.0,
                    help="cup speed relative to the saucer (px/s) below which the cup counts as at rest")
    ap.add_argument("--min-move-s", type=float, default=1.0, help="min duration of a rest")
    ap.add_argument("--merge-gap-s", type=float, default=1.5, help="merge moving runs closer than this")
    ap.add_argument("--window-s", type=float, default=3.0, help="scoring window at the start of the ending rest")
    args = ap.parse_args()

    seg = json.loads((ROOT / args.seg_dir / f"{args.episode}.json").read_text())
    frames = seg["frames"]
    keys = sorted(frames, key=int)
    src = seg["source_indices"]
    fps = float(seg.get("fps") or 30.0)
    stride = int(seg.get("stride") or 1)
    dt = stride / fps                              # seconds between sampled frames
    t = np.array([src[int(k)] for k in keys]) / fps

    cents = [cup_centroid(frames[k]) for k in keys]
    xy = np.array([c if c is not None else [np.nan, np.nan] for c in cents], dtype=float)
    # Head-mounted camera: the cup's image position moves whenever the head moves, even
    # with the cup sitting still. Measure cup motion RELATIVE to the saucer — a static object
    # in the scene — so camera motion cancels. Falls back to the raw centroid only where no
    # saucer is visible.
    sc = []
    for k in keys:
        b = saucer_box(frames[k], pad=0.0)
        sc.append([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] if b else [np.nan, np.nan])
    sc = np.array(sc, dtype=float)
    rel = xy - sc
    d_rel = np.linalg.norm(np.diff(rel, axis=0), axis=1) / dt
    d_raw = np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt
    d = np.where(np.isfinite(d_rel), d_rel, np.nan)   # no saucer -> unknown, treated as still
    speed = np.concatenate(([np.nan], d))
    still = np.nan_to_num(speed, nan=0.0) < args.speed_thresh

    # per-frame SEG indicator: cup centroid inside per-frame saucer box (carry last box)
    inside = np.full(len(keys), np.nan)
    last_box = None
    for i, k in enumerate(keys):
        b = saucer_box(frames[k])
        if b is not None:
            last_box = b
        c = cents[i]
        if c is not None and last_box is not None:
            inside[i] = float(last_box[0] <= c[0] <= last_box[2] and last_box[1] <= c[1] <= last_box[3])

    # VLM trace mapped by source frame index (strides may differ)
    p_done = None
    vlm_label = "—"
    if args.vlm_dir:
        vp = ROOT / args.vlm_dir / f"{args.episode}.json"
        if vp.exists():
            v = json.loads(vp.read_text())
            vs = np.array(v["source_indices"]); vp_ = np.array(v["p_yes"], dtype=float)
            p_done = np.interp(np.array([src[int(k)] for k in keys]), vs, vp_)
            vlm_label = v.get("model", args.vlm_dir)

    # Rests: the cup sitting still (relative to the saucer) for >= min_rest. Each rest is
    # ON the saucer or OFF it. A person doing this task repeatedly produces
    #   off -> on   (an attempt that succeeded)
    #   off -> off  (an attempt that ended with the cup set down elsewhere: failure)
    #   on  -> off  (a return to the start position: not an attempt)
    min_rest = max(1, int(round(args.min_move_s / dt)))
    rests = [(a, b) for a, b in runs(still) if b - a >= min_rest]
    rest_on = []
    for a, b in rests:
        seg_v = inside[a:b]
        rest_on.append(bool(np.isfinite(seg_v).any() and np.nanmean(seg_v) >= 0.5))
    win = max(1, int(round(args.window_s / dt)))
    rows, returns = [], 0
    for i in range(len(rests) - 1):
        (a0, b0), (a1, b1) = rests[i], rests[i + 1]
        if rest_on[i]:
            returns += 1          # started on the saucer: a return, not an attempt
            continue
        w0, w1 = a1, min(b1, a1 + win)
        seg_score = float(np.nanmean(inside[w0:w1])) if np.isfinite(inside[w0:w1]).any() else None
        vlm_score = float(np.mean(p_done[w0:w1])) if p_done is not None else None
        avail = [x for x in (seg_score, vlm_score) if x is not None]
        tag = ("success" if np.mean(avail) >= 0.5 else "failure") if avail else "—"
        rows.append({"attempt": len(rows) + 1, "start_s": t[b0 - 1], "settle_s": t[a1],
                     "duration_s": (a1 - b0 + 1) * dt, "seg": seg_score, "vlm": vlm_score, "tag": tag,
                     "start_frame": int(src[int(keys[b0 - 1])]), "settle_frame": int(src[int(keys[a1])])})
    attempts = [(max(0, int(round((r["start_s"]) / dt))), int(round(r["settle_s"] / dt))) for r in rows]

    # ---- report
    OUT.mkdir(exist_ok=True)
    ns = sum(r["tag"] == "success" for r in rows); nf = sum(r["tag"] == "failure" for r in rows)
    md = [f"# {args.episode} — split into attempts\n",
          f"{len(keys)} sampled frames at {1/dt:.1f} fps ({t[-1]:.0f} s). Rest = cup still relative to "
          f"the saucer (< {args.speed_thresh:.0f} px/s) for ≥ {args.min_move_s}s. Attempt = a rest OFF the "
          f"saucer to the next rest; success if that rest is ON the saucer (SEG) and the VLM agrees over "
          f"its first {args.window_s}s. Transitions that start ON the saucer are returns to the start "
          f"position ({returns} of them), not attempts. VLM: {vlm_label}.\n",
          f"**{len(rows)} attempts: {ns} success / {nf} failure**  ·  {returns} returns  ·  "
          f"{len(rests)} rests ({sum(rest_on)} on saucer)\n",
          "| # | start (s) | settle (s) | carry (s) | SEG cup-in-saucer | VLM p(done) | tag |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        f = lambda x: "—" if x is None else f"{x:.2f}"
        md.append(f"| {r['attempt']} | {r['start_s']:.1f} | {r['settle_s']:.1f} | {r['duration_s']:.1f} | "
                  f"{f(r['seg'])} | {f(r['vlm'])} | {'✅' if r['tag']=='success' else '❌' if r['tag']=='failure' else '—'} |")
    text = "\n".join(md)
    (OUT / f"aria_split_{args.episode}.md").write_text(text + "\n")
    print(text)

    # ---- plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(12, 6.5), sharex=True)
    for ax in axes:
        for s, e in attempts:
            ax.axvspan(t[s], t[min(e, len(t) - 1)], color="#1982c4", alpha=0.12)
        for r in rows:
            ax.axvline(r["settle_s"], color="#8ac926" if r["tag"] == "success" else "#ff595e", lw=1.2, ls="--")
    axes[0].plot(t, np.nan_to_num(speed), color="#333", lw=1); axes[0].axhline(args.speed_thresh, color="#999", ls=":")
    axes[0].set_ylabel("cup speed rel. saucer px/s")
    axes[1].plot(t, inside, color="#ffca3a", lw=1.4, drawstyle="steps-post"); axes[1].set_ylim(-0.05, 1.05); axes[1].set_ylabel("cup in saucer")
    if p_done is not None:
        axes[2].plot(t, p_done, color="#1982c4", lw=1.4)
    axes[2].axhline(0.5, color="#999", ls=":"); axes[2].set_ylim(-0.02, 1.02); axes[2].set_ylabel("VLM p(done)")
    axes[2].set_xlabel("time (s)")
    fig.suptitle(f"{args.episode}: {len(rows)} attempts — {ns} success / {nf} failure  (shaded = attempt: pick-up to set-down; dashed = set-down, coloured by tag)")
    fig.tight_layout(); fig.savefig(OUT / f"aria_split_{args.episode}.png", dpi=130)
    print(f"\n✓ wrote results/aria_split_{args.episode}.md and .png")


if __name__ == "__main__":
    main()
