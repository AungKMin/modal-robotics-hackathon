#!/usr/bin/env python3
"""
One episode, the whole critic: SAM 3 + Qwen3-VL + PaliGemma 2 + Cosmos-Reason2 → final verdict.

    python3 run_episode.py 692e98927641010d04354574
    python3 run_episode.py 2025-11-24-23-59-28-546000 --volume egoverse-aria --stride 15
    python3 run_episode.py 692ea3be --volume egoverse-cup50-full --stride 10 --no-render

Runs the four Modal apps in parallel (each is its own `modal run`), then fuses the traces:

    SEG   = SAM 3 cup-centroid-in-saucer-box over the last quarter of frames
    VLM_m = p("Yes") to "is the cup resting on the saucer?" over the last quarter, per model
    final = mean(SEG, mean(VLM_m))  →  success if >= 0.5

Success = the cup ends up on the saucer; failure = it does not.

The episode argument is a name PREFIX (the apps' --match); it must resolve to exactly one
episode in the Volume. Outputs land under run_episode_out/<episode>/:
    sam3.json, sam3_overlay.mp4, <model>.json, <model>_meter.mp4, verdict.md
Nothing here is new — it drives sam3/episodes.py and vlm_critic/app.py exactly as the batch
runs do, so a verdict from this script is reproducible from those apps alone.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "fuse"))
from fuse import seg_features, vlm_features  # noqa: E402

MODELS = ["qwen", "paligemma", "cosmos"]
CUP_QUESTION = "Is the cup resting on the saucer? Answer Yes or No."


def sh(cmd, env, log):
    with open(log, "w") as fh:
        return subprocess.Popen(cmd, cwd=HERE, env=env, stdout=fh, stderr=subprocess.STDOUT)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode", help="episode name or unique prefix")
    ap.add_argument("--volume", default="egoverse-cup50-full", help="Modal Volume holding the episodes")
    ap.add_argument("--stride", type=int, default=10, help="frame stride (30 fps source: 10 = 3 fps)")
    ap.add_argument("--prompts", default="hand,cup,saucer", help="SAM 3 prompts")
    ap.add_argument("--gpu", default="H100")
    ap.add_argument("--no-render", action="store_true", help="skip overlay / meter videos")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--out", default="run_episode_out")
    args = ap.parse_args()

    models = [m for m in args.models.split(",") if m]
    tmp = HERE / f".run_episode_tmp_{args.episode}"
    tmp.mkdir(exist_ok=True)
    env = dict(os.environ, EPISODES_VOLUME=args.volume, VLM_GPU=args.gpu, VLM_CONTAINERS="1", VLM_BATCH="16")
    render = [] if not args.no_render else ["--no-render"]

    procs = {}
    procs["sam3"] = sh(
        ["modal", "run", "sam3/episodes.py", "--match", args.episode, "--stride", str(args.stride),
         "--prompts", args.prompts, "--out", str(tmp / "sam3")] + render, env, tmp / "sam3.log")
    def _vlm_cmd(m):
        cmd = ["modal", "run", "vlm_critic/app.py", "--model", m, "--match", args.episode,
               "--stride", str(args.stride), "--out", str(tmp / "vlm")] + render
        if m == "paligemma":
            cmd += ["--question", CUP_QUESTION]   # short literal question for the 3B VQA model
        return cmd
    for m in models:
        procs[m] = sh(_vlm_cmd(m), env, tmp / f"{m}.log")

    vlm_cmd = _vlm_cmd

    def wait(procs):
        while any(p.poll() is None for p in procs.values()):
            time.sleep(5)

    def produced(name):
        # Success is "a trace exists", not the exit code: a successful modal run can still exit
        # non-zero from the client's teardown noise, and a transient scheduling error can kill
        # one leg while the others finish.
        d = tmp / ("sam3" if name == "sam3" else f"vlm/{name}")
        return any(d.glob("*.json"))

    print(f"launched {len(procs)} Modal runs for '{args.episode}' on {args.volume} (stride {args.stride}) …")
    t0 = time.time()
    wait(procs)
    for name, p in procs.items():
        print(f"  {name:10s} {'ok' if produced(name) else f'no trace (exit {p.returncode})'}")

    # one retry for any leg that produced nothing
    retry = {}
    for name in procs:
        if produced(name):
            continue
        print(f"  retrying {name} …")
        if name == "sam3":
            retry[name] = sh(["modal", "run", "sam3/episodes.py", "--match", args.episode, "--stride", str(args.stride),
                              "--prompts", args.prompts, "--out", str(tmp / "sam3")] + render, env, tmp / "sam3.retry.log")
        else:
            retry[name] = sh(vlm_cmd(name), env, tmp / f"{name}.retry.log")
    if retry:
        wait(retry)
        for name in retry:
            print(f"  {name:10s} {'ok on retry' if produced(name) else 'FAILED twice — see ' + str(tmp / (name + '.retry.log'))}")
    print(f"all done in {time.time() - t0:.0f}s")

    # ---- collect
    seg_files = list((tmp / "sam3").glob("*.json"))
    if len(seg_files) != 1:
        sys.exit(f"expected exactly one SAM 3 trace for prefix '{args.episode}', found {len(seg_files)} "
                 f"— make the prefix unique")
    seg = json.loads(seg_files[0].read_text())
    ep = seg["episode"]
    out = HERE / args.out / ep
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(seg_files[0], out / "sam3.json")
    for v in (tmp / "sam3").glob(f"{ep}_overlay.mp4"):
        shutil.copy(v, out / "sam3_overlay.mp4")

    sf = seg_features(seg)
    seg_score = sf.get("seg_score")
    vs, top = {}, {}
    for m in models:
        f = tmp / "vlm" / m / f"{ep}.json"
        if f.exists():
            rec = json.loads(f.read_text())
            shutil.copy(f, out / f"{m}.json")
            mv = tmp / "vlm" / m / f"{ep}_meter.mp4"
            if mv.exists():
                shutil.copy(mv, out / f"{m}_meter.mp4")
            vs[m] = vlm_features(rec)["p_done_late"]
            top[m] = rec.get("last_frame_top_tokens")

    parts = ([seg_score] if seg_score is not None else []) + ([sum(vs.values()) / len(vs)] if vs else [])
    final = sum(parts) / len(parts) if parts else None
    tag = None if final is None else ("success" if final >= 0.5 else "failure")
    votes = ([("SEG", seg_score >= 0.5)] if seg_score is not None else []) + [(m, v >= 0.5) for m, v in vs.items()]

    f2 = lambda x: "—" if x is None else f"{x:.2f}"
    md = [f"# {ep} — verdict\n",
          f"Volume `{args.volume}`, stride {args.stride}, {len(seg['frames'])} frames scored. "
          f"Success = the cup ends up on the saucer.\n",
          f"## **{'✅ SUCCESS' if tag == 'success' else '❌ FAILURE' if tag else '—'}**  (final {f2(final)}, "
          f"{sum(v for _, v in votes)}/{len(votes)} signals say success)\n",
          "| signal | score | vote |", "|---|---|---|",
          f"| SAM 3 cup-in-saucer ({sf.get('seg_mode', '—')}) | {f2(seg_score)} | {'✅' if seg_score is not None and seg_score >= 0.5 else '❌' if seg_score is not None else '—'} |"]
    for m in models:
        md.append(f"| {m} p(cup on saucer), last quarter | {f2(vs.get(m))} | "
                  f"{'✅' if m in vs and vs[m] >= 0.5 else '❌' if m in vs else '—'} |")
    md.append(f"| **final = mean(SEG, mean VLM)** | **{f2(final)}** | **{tag or '—'}** |")
    md.append("\nLast-frame top tokens (what each model actually wanted to say):")
    for m, t in top.items():
        if t:
            md.append(f"- {m}: " + ", ".join(f"`{tok}`={p}" for tok, p in t[:3]))
    md.append(f"\nFiles: {out}/")
    text = "\n".join(md)
    (out / "verdict.md").write_text(text + "\n")
    print("\n" + text)
    missing = [m for m in models if m not in vs]
    for lg in tmp.glob("*.log"):
        shutil.copy(lg, out / lg.name)          # keep the Modal logs next to the verdict
    if missing:
        print(f"\n⚠ no trace for: {missing} — logs kept in {out}/")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
