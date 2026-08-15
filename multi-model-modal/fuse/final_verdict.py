"""
Final success/failure determination per episode: SAM 3 geometry + every VLM available.

    python3 fuse/final_verdict.py --seg-dir sam3_out/cup50 --vlm-root vlm_critic_out_cup50 --name cup50

Definition: SUCCESS = the cup ends up resting on the saucer; FAILURE = it does not.
Both signals implement exactly that: SEG tests cup-centroid-in-saucer-box over the last quarter of
frames, the VLMs are asked "is the cup resting on the saucer?" and read out as p(yes).

Per episode:
  SEG   = fraction of last-quarter frames with the cup centroid inside the (per-frame) saucer box
  VLM_m = p(done) over the last quarter, per model m present for that episode
  final = mean(SEG, mean(VLM_m))    -> success if >= 0.5
Both halves weighted equally so a confident VLM cannot outvote the geometry on its own and vice
versa. Where only SEG exists (VLMs were run on a subset), the verdict is SEG-only and marked so.
Writes results/<name>_final.md.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from fuse import seg_features, vlm_features, label_of  # noqa

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT.parent / "results"
ap = argparse.ArgumentParser()
ap.add_argument("--seg-dir", required=True); ap.add_argument("--vlm-root", required=True)
ap.add_argument("--name", default="cup50"); ap.add_argument("--stat", default="late", choices=["late", "max"])
a = ap.parse_args()

def load(d):
    d = ROOT / d
    return {p.stem: json.loads(p.read_text()) for p in sorted(d.glob("*.json"))} if d.is_dir() else {}

seg = load(a.seg_dir)
vlms = {m.name: load(f"{a.vlm_root}/{m.name}") for m in sorted((ROOT / a.vlm_root).iterdir()) if m.is_dir()}
models = [m for m in vlms if vlms[m]]
episodes = sorted(set(seg) | {e for m in models for e in vlms[m]})

rows = []
for ep in episodes:
    sf = seg_features(seg[ep]) if ep in seg else {}
    seg_s = sf.get("seg_score")
    vs = {}
    for m in models:
        if ep in vlms[m] and vlms[m][ep].get("p_yes"):
            vf = vlm_features(vlms[m][ep]); vs[m] = vf["p_done_max"] if a.stat == "max" else vf["p_done_late"]
    parts = []
    if seg_s is not None: parts.append(seg_s)
    if vs: parts.append(float(np.mean(list(vs.values()))))
    final = float(np.mean(parts)) if parts else None
    tag = None if final is None else ("success" if final >= 0.5 else "failure")
    votes = ([("SEG", seg_s >= 0.5)] if seg_s is not None else []) + [(m, v >= 0.5) for m, v in vs.items()]
    agree = f"{sum(v for _, v in votes)}/{len(votes)}" if votes else "—"
    lab = label_of(seg.get(ep) or next((vlms[m][ep] for m in models if ep in vlms[m]), {}))
    rows.append((ep, lab, seg_s, vs, final, tag, agree))

n_full = sum(1 for r in rows if r[3]); ns = sum(r[5] == "success" for r in rows); nf = sum(r[5] == "failure" for r in rows)
md = [f"# {a.name} — final verdict (SAM 3 + VLMs)\n",
      f"**Success = the cup ends up on the saucer; failure = it does not.** {len(rows)} episodes · {n_full} with VLM scores ({', '.join(models)}) · rest SEG-only. "
      f"final = mean(SEG, mean(VLMs)) ≥ 0.5. Statistic: `{a.stat}`.\n",
      f"**Final: {ns} success / {nf} failure → {100*nf/max(1,ns+nf):.0f}% failure prevalence**\n"]
sub = [r for r in rows if r[3]]
if sub:
    s2 = sum(r[5]=="success" for r in sub); md.append(f"On the {len(sub)} episodes with all signals: {s2} success / {len(sub)-s2} failure.\n")
hdr = ["episode", "label", "SEG"] + models + ["final", "votes ✅", "verdict"]
md.append("| " + " | ".join(hdr) + " |"); md.append("|" + "---|" * len(hdr))
f = lambda x: "—" if x is None else f"{x:.2f}"
for ep, lab, seg_s, vs, final, tag, agree in rows:
    md.append("| " + " | ".join([ep, lab or "—", f(seg_s)] + [f(vs.get(m)) for m in models] +
              [f(final), agree, ("✅ success" if tag == "success" else "❌ failure" if tag else "—") + ("" if vs else " (SEG only)")]) + " |")
text = "\n".join(md); OUT.mkdir(exist_ok=True); (OUT / f"{a.name}_final.md").write_text(text + "\n"); print(text)
