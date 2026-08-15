"""
Client for the deployed SAM 3 concept segmenter.

Smoke test (downloads a sample clip, tracks "person"):
    python sam3/try.py

Real use — a text prompt parsed from an episode's language annotation:
    python sam3/try.py --video episode.mp4 --prompt "the towel" --prompt "hand"
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

import modal
from rich import print
from rich.console import Console
from rich.table import Table

SAMPLE_VIDEO = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/bedroom.mp4"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAM 3 concept segmentation on a video.")
    parser.add_argument("--video", default=None, help="Path to an mp4. Omit to use the sample clip.")
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="Noun phrase to track. Repeat for multiple concepts.",
    )
    parser.add_argument("--max-frames", type=int, default=50, help="Cap frames tracked.")
    parser.add_argument("--out", default="sam3_trace.json", help="Where to write the per-frame trace.")
    args = parser.parse_args()

    prompts = args.prompt or ["person"]

    if args.video:
        video_bytes = Path(args.video).read_bytes()
        source = args.video
    else:
        print(f"[dim]No --video given, fetching sample clip[/dim]")
        video_bytes = urllib.request.urlopen(SAMPLE_VIDEO).read()
        source = SAMPLE_VIDEO

    segmenter = modal.Cls.from_name("sam3-concept-segmentation", "ConceptSegmenter")()

    print("\n")
    print("-" * 80)
    print(f"Running {Path(__file__).name} against the deployed SAM 3 app")
    print(f"Source:  {source}")
    print(f"Prompts: {prompts}")
    print("-" * 80)

    console = Console()
    start = time.perf_counter()
    with console.status(
        (
            "[green]Loading SAM 3 on a cloud GPU and propagating masks through the clip.[/green]\n"
            f"[green]View progress in Modal dashboard: "
            f"[magenta]{segmenter.segment.get_dashboard_url()}[/magenta][/green]"
        ),
        spinner="dots",
    ):
        result = segmenter.segment.remote(
            prompts=prompts,
            video_bytes=video_bytes,
            max_frames=args.max_frames,
        )
    elapsed = time.perf_counter() - start

    frames = result["frames"]
    print(f"[green]Elapsed: {elapsed:.2f}s for {result['num_frames']} frames "
          f"({elapsed / max(result['num_frames'], 1):.2f}s/frame)[/green]")

    table = Table(title="Detections (first 10 frames)")
    for col in ("frame", "obj id", "score", "area px", "centroid x,y"):
        table.add_column(col)
    for frame_idx in sorted(frames, key=int)[:10]:
        for obj in frames[frame_idx]["objects"]:
            centroid = obj["centroid_xy"]
            table.add_row(
                str(frame_idx),
                str(obj["object_id"]),
                f"{obj['score']:.3f}",
                str(obj["area"]),
                f"{centroid[0]:.0f},{centroid[1]:.0f}" if centroid else "-",
            )
    console.print(table)

    # Object identity is stable across frames, so this is the count of distinct tracks.
    track_ids = {o["object_id"] for f in frames.values() for o in f["objects"]}
    print(f"[green]{len(track_ids)} distinct tracks: {sorted(track_ids)}[/green]")

    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"[green]✓ Per-frame trace written to: {args.out}[/green]")


if __name__ == "__main__":
    main()
