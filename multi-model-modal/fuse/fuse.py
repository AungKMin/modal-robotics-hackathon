"""
Fusion + eval: turn the per-episode traces into a success/failure tag and score it.

    python3 fuse/fuse.py                       # uses vlm_critic_out/qwen by default
    python3 fuse/fuse.py --model cosmos

Reads whatever exists of:
    vlm_critic_out/<model>/<ep>.json    p(done) trace          (VLM critic)
    sam3_out/episodes/<ep>.json         cup / saucer tracks    (SAM 3)
    geometric_out/<ep>.json             holds, handover        (proprioception)

and evaluates each criterion alone, then fused, against the labels already carried in every
trace (task_name ends in _success / _failure; the slice is 10/10).

Criteria — task is "pick up cup, hand over, place it on the saucer":

  [VLM]  p_done_late  = mean p(done) over the last quarter of sampled frames.
         success if >= 0.5. A task is judged by how it ended, so the late window, not the
         mean over the whole episode (every episode starts incomplete).

  [SEG]  cup_on_saucer = fraction of last-quarter frames in which the cup centroid lies
         inside the saucer box (padded 25%). success if >= 0.5. If no saucer track exists
         the fallback is cup_settled: cup detected at the end AND its centroid moved from
         where it started (it was picked up and put down somewhere) — weaker, and flagged.

  [GEO]  any_hold: some gripper actually closed on the object for >= 0.3 s. Not sufficient
         for success but necessary — an episode with no hold cannot have succeeded. Used
         as a veto, not a vote.

  [FUSED] score = mean of the available continuous signals (p_done_late, cup_on_saucer),
          zeroed if GEO vetoes. success if >= 0.5.

Numpy only. Writes fuse_out/results.json and fuse_out/summary.md (paste into the slide).
"""

import argparse
import json
from pathlib import Path

import numpy as np

SAUCER_WORDS = ("saucer", "plate", "tray", "dish")
CUP_WORDS = ("cup", "mug")


def load_dir(d: Path) -> dict:
    out = {}
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*.json")):
        try:
            out[p.stem] = json.loads(p.read_text())
        except Exception as e:  # noqa: BLE001
            print(f"  skip {p}: {e}")
    return out


def label_of(rec: dict):
    lab = rec.get("label")
    if lab in ("success", "failure"):
        return lab
    tn = rec.get("task_name") or ""
    return "success" if tn.endswith("_success") else "failure" if tn.endswith("_failure") else None


# ---------------------------------------------------------------- VLM -------------------
def vlm_features(rec: dict) -> dict:
    tr = rec.get("p_yes") or []
    if not tr:
        return {}
    q = max(1, len(tr) // 4)
    early, late = float(np.mean(tr[:q])), float(np.mean(tr[-q:]))
    return {
        "p_done_late": late,
        "p_done_final": float(tr[-1]),
        "p_done_max": float(max(tr)),
        "p_done_rise": late - early,
    }


# ---------------------------------------------------------------- SEG -------------------
def _tracks_for(frame: dict, words) -> list:
    ids = set()
    for prompt, obj_ids in (frame.get("prompt_to_obj_ids") or {}).items():
        if any(w in prompt.lower() for w in words):
            ids.update(int(i) for i in obj_ids)
    return [o for o in frame.get("objects", []) if int(o["object_id"]) in ids]


def seg_features(rec: dict) -> dict:
    frames = rec.get("frames") or {}
    if not frames:
        return {}
    keys = sorted(frames, key=int)
    n = len(keys)
    q = max(1, n // 4)
    last, first = keys[-q:], keys[:q]
    W = 640.0  # frame width for normalising displacement; overlay frames are 480x640

    # Was a saucer ever tracked? If so, use cup-in-saucer geometry.
    saucer_seen = any(_tracks_for(frames[k], SAUCER_WORDS) for k in keys)

    def cup_centroid(k):
        cups = _tracks_for(frames[k], CUP_WORDS)
        cups = [c for c in cups if c.get("centroid_xy")]
        if not cups:
            return None
        c = max(cups, key=lambda o: o["score"])  # the most confident cup
        return np.array(c["centroid_xy"], dtype=float)

    def saucer_box(k):
        s = _tracks_for(frames[k], SAUCER_WORDS)
        if not s:
            return None
        s = max(s, key=lambda o: o["score"])
        x0, y0, x1, y1 = s["box_xyxy"]
        pw, ph = 0.25 * (x1 - x0), 0.25 * (y1 - y0)
        return (x0 - pw, y0 - ph, x1 + pw, y1 + ph)

    feats = {"saucer_tracked": bool(saucer_seen)}
    cup_end_present = np.mean([cup_centroid(k) is not None for k in last])
    feats["cup_present_end"] = float(cup_end_present)

    if saucer_seen:
        # Saucer is static: use its box from any frame where it was seen (median box).
        boxes = [saucer_box(k) for k in keys if saucer_box(k) is not None]
        bx = np.median(np.array(boxes), axis=0)
        inside = []
        for k in last:
            c = cup_centroid(k)
            if c is None:
                inside.append(0.0)
                continue
            inside.append(float(bx[0] <= c[0] <= bx[2] and bx[1] <= c[1] <= bx[3]))
        feats["cup_on_saucer"] = float(np.mean(inside))
        feats["seg_score"] = feats["cup_on_saucer"]
        feats["seg_mode"] = "cup_on_saucer"
    else:
        # Fallback: cup ended up somewhere else than it started, and is still visible.
        c0 = [cup_centroid(k) for k in first]
        c1 = [cup_centroid(k) for k in last]
        c0 = [c for c in c0 if c is not None]
        c1 = [c for c in c1 if c is not None]
        if c0 and c1:
            disp = float(np.linalg.norm(np.mean(c1, axis=0) - np.mean(c0, axis=0)) / W)
        else:
            disp = 0.0
        feats["cup_displacement_frac"] = disp
        # settled = present at end AND moved at least ~8% of the frame width
        feats["cup_settled"] = float(cup_end_present >= 0.5 and disp >= 0.08)
        feats["seg_score"] = min(1.0, disp / 0.16) * (1.0 if cup_end_present >= 0.5 else 0.0)
        feats["seg_mode"] = "cup_settled(fallback)"

    # identity churn: many distinct cup ids = tracking lost the object
    cup_ids = set()
    for k in keys:
        for o in _tracks_for(frames[k], CUP_WORDS):
            cup_ids.add(int(o["object_id"]))
    feats["cup_track_ids"] = len(cup_ids)
    return feats


# ---------------------------------------------------------------- GEO -------------------
def geo_features(rec: dict) -> dict:
    f = rec.get("features") or {}
    bi = f.get("bimanual") or {}
    return {
        "any_hold": bool(bi.get("any_hold", False)),
        "handover": bool(bi.get("handover_detected", False)),
        "total_slips": int(bi.get("total_slips") or 0),
    }


# ---------------------------------------------------------------- eval ------------------
def auroc(scores: list, labels: list) -> float:
    """Mann-Whitney AUROC; ties count half. labels: 1 = success."""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def acc(pred: list, labels: list) -> float:
    return float(np.mean([p == l for p, l in zip(pred, labels)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen", help="vlm_critic_out/<model>")
    ap.add_argument("--vlm-thresh", type=float, default=0.5)
    ap.add_argument("--seg-thresh", type=float, default=0.5)
    ap.add_argument("--out", default="fuse_out")
    args = ap.parse_args()

    vlm = load_dir(Path("vlm_critic_out") / args.model)
    seg = load_dir(Path("sam3_out") / "episodes")
    geo = load_dir(Path("geometric_out"))
    print(f"loaded: vlm={len(vlm)} ({args.model})  seg={len(seg)}  geo={len(geo)}")

    episodes = sorted(set(vlm) | set(seg) | set(geo))
    rows = []
    for ep in episodes:
        rec = vlm.get(ep) or seg.get(ep) or geo.get(ep)
        lab = label_of(rec)
        row = {"episode": ep, "label": lab}
        row.update({f"vlm.{k}": v for k, v in vlm_features(vlm.get(ep, {})).items()})
        row.update({f"seg.{k}": v for k, v in seg_features(seg.get(ep, {})).items()})
        row.update({f"geo.{k}": v for k, v in geo_features(geo.get(ep, {})).items()})

        # per-criterion predictions
        if "vlm.p_done_late" in row:
            row["pred.vlm"] = "success" if row["vlm.p_done_late"] >= args.vlm_thresh else "failure"
        if "seg.seg_score" in row:
            row["pred.seg"] = "success" if row["seg.seg_score"] >= args.seg_thresh else "failure"
        # fused: mean of available continuous signals; GEO veto
        sig = [row[k] for k in ("vlm.p_done_late", "seg.seg_score") if k in row]
        if sig:
            score = float(np.mean(sig))
            if "geo.any_hold" in row and not row["geo.any_hold"]:
                score = 0.0
            row["fused.score"] = score
            row["pred.fused"] = "success" if score >= 0.5 else "failure"
        rows.append(row)

    labelled = [r for r in rows if r["label"]]
    y = [1 if r["label"] == "success" else 0 for r in labelled]

    # ---- report
    lines = []
    lines.append(f"# Fusion eval — {len(labelled)} labelled episodes "
                 f"({sum(y)} success / {len(y)-sum(y)} failure), VLM={args.model}\n")
    lines.append("| criterion | n | accuracy | AUROC | note |")
    lines.append("|---|---|---|---|---|")
    for name, pred_key, score_key, note in [
        ("VLM  p_done_late >= %.2f" % args.vlm_thresh, "pred.vlm", "vlm.p_done_late", "logit-derived, late window"),
        ("SEG  cup-on-saucer >= %.2f" % args.seg_thresh, "pred.seg", "seg.seg_score",
         "fallback=cup_settled where no saucer track"),
        ("FUSED mean(VLM,SEG) w/ GEO veto", "pred.fused", "fused.score", "any_hold=False -> failure"),
    ]:
        sub = [r for r in labelled if pred_key in r]
        if not sub:
            lines.append(f"| {name} | 0 | — | — | no outputs found |")
            continue
        yy = [1 if r["label"] == "success" else 0 for r in sub]
        a = acc([r[pred_key] for r in sub], [r["label"] for r in sub])
        u = auroc([r[score_key] for r in sub], yy)
        lines.append(f"| {name} | {len(sub)} | {a:.2f} | {u:.2f} | {note} |")

    if any("geo.any_hold" in r for r in labelled):
        veto = [r for r in labelled if "geo.any_hold" in r and not r["geo.any_hold"]]
        lines.append(f"\nGEO veto fired on {len(veto)} episode(s): "
                     + ", ".join(f"{r['episode']}({r['label']})" for r in veto))
    modes = {r.get("seg.seg_mode") for r in labelled if "seg.seg_mode" in r}
    if modes:
        lines.append(f"SEG mode(s) in use: {sorted(m for m in modes if m)}"
                     + ("  ← re-run sam3 with --prompts \"cup,saucer\" for the geometric criterion"
                        if any("fallback" in (m or "") for m in modes) else ""))

    lines.append("\n## Per-episode\n")
    lines.append("| episode | label | p_done_late | seg_score | any_hold | fused | pred | ok |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in labelled:
        def f(k, fmt="{:.2f}"):
            return fmt.format(r[k]) if k in r and r[k] is not None else "—"
        pred = r.get("pred.fused") or r.get("pred.vlm") or r.get("pred.seg") or "—"
        ok = "✓" if pred == r["label"] else ("✗" if pred != "—" else "")
        lines.append(f"| {r['episode']} | {r['label']} | {f('vlm.p_done_late')} | {f('seg.seg_score')} | "
                     f"{r.get('geo.any_hold', '—')} | {f('fused.score')} | {pred} | {ok} |")

    text = "\n".join(lines)
    print("\n" + text)
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps(rows, indent=1))
    (out / "summary.md").write_text(text + "\n")
    print(f"\n✓ wrote {out}/results.json and {out}/summary.md")


if __name__ == "__main__":
    main()
