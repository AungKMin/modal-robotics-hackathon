"""
Render SAM 3 mask overlays from a trace produced by `modal run sam3/app.py --return-masks`.

    python3 sam3/visualize.py --all --video    # overlay mp4 per clip
    python3 sam3/visualize.py --all            # contact-sheet png per clip
    python3 sam3/visualize.py --trace sam3_out/tennis.json --video

Two output modes. `--video` re-encodes every traced frame with its masks burned in, at the
source clip's frame rate — this is the one to put in a demo. Without it you get a contact
sheet: N sampled frames tiled into a grid, which is faster to skim when checking many clips.

Either way each object's mask is tinted in its own colour with the bounding box and
`id:score` label on top. Object colour is keyed to the SAM 3 object id, so a track keeps its
colour for its whole life — if a colour jumps, identity was lost, which is exactly the
failure you want to catch by eye.

Requires masks. A trace written without --return-masks carries only boxes and centroids;
this script says so and falls back to drawing boxes alone.

Dependencies (local, not in the Modal image):
    uv pip install numpy pillow
Frame decoding shells out to ffmpeg, which you already have.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import numpy as np
    from PIL import Image, ImageDraw
except ImportError:
    sys.exit(
        "This script needs numpy and pillow locally:\n"
        "    uv pip install numpy pillow\n"
        "(The Modal image has them; your local env does not.)"
    )

# Qualitative palette — distinct in hue and in luminance, so overlapping tints stay
# tellable apart and the labels stay readable on any of them.
PALETTE = [
    (255, 89, 94),
    (56, 176, 0),
    (25, 130, 196),
    (255, 202, 58),
    (154, 78, 174),
    (255, 146, 76),
    (0, 187, 249),
    (241, 91, 181),
]
ALPHA = 0.45


def color_for(obj_id: int) -> tuple:
    return PALETTE[obj_id % len(PALETTE)]


def extract_frames(video: Path, frame_indices: list[int], workdir: Path) -> dict:
    """Pull specific frame numbers out of a clip with ffmpeg. Returns {idx: PIL.Image}."""
    wanted = "+".join(f"eq(n\\,{i})" for i in frame_indices)
    out_pattern = workdir / "f%04d.png"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-i", str(video),
            "-vf", f"select='{wanted}'",
            "-vsync", "0",
            str(out_pattern),
        ],
        check=True,
    )
    # ffmpeg numbers the *selected* frames 1..N in order, not by source index.
    written = sorted(workdir.glob("f*.png"))
    if len(written) != len(frame_indices):
        print(
            f"  warning: asked for {len(frame_indices)} frames, ffmpeg wrote {len(written)}",
            file=sys.stderr,
        )
    return {idx: Image.open(p).convert("RGB") for idx, p in zip(frame_indices, written)}


def load_mask(bin_path: Path, obj: dict) -> "np.ndarray | None":
    """Unpack one object's bit-packed mask back to a bool array at video resolution."""
    if "mask_offset" not in obj:
        return None
    height, width = obj["mask_shape"]
    with open(bin_path, "rb") as fh:
        fh.seek(obj["mask_offset"])
        raw = fh.read(obj["mask_nbytes"])
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
    return bits[: height * width].reshape(height, width).astype(bool)


def draw_panel(frame: "Image.Image", objects: list, bin_path, frame_idx: int) -> "Image.Image":
    """Tint each mask, outline each box, label each track."""
    base = frame.convert("RGBA")
    arr = np.array(base, dtype=np.float32)

    for obj in objects:
        mask = load_mask(bin_path, obj) if bin_path else None
        if mask is None:
            continue
        if mask.shape != arr.shape[:2]:
            print(
                f"  warning: mask {mask.shape} != frame {arr.shape[:2]}, skipping obj "
                f"{obj['object_id']}",
                file=sys.stderr,
            )
            continue
        tint = np.array(color_for(obj["object_id"]), dtype=np.float32)
        arr[mask, :3] = arr[mask, :3] * (1 - ALPHA) + tint * ALPHA

    out = Image.fromarray(arr.astype("uint8"), "RGBA")
    draw = ImageDraw.Draw(out)

    for obj in objects:
        x0, y0, x1, y1 = obj["box_xyxy"]
        color = color_for(obj["object_id"])
        draw.rectangle([x0, y0, x1, y1], outline=color + (255,), width=3)
        label = f"{obj['object_id']}:{obj['score']:.2f}"
        tx, ty = x0 + 3, max(0, y0 - 14)
        draw.rectangle([tx - 2, ty - 1, tx + 7 * len(label), ty + 12], fill=(0, 0, 0, 190))
        draw.text((tx, ty), label, fill=color + (255,))

    draw.text((6, 6), f"frame {frame_idx}", fill=(255, 255, 255, 255))
    draw.text((5, 5), f"frame {frame_idx}", fill=(0, 0, 0, 255))
    return out.convert("RGB")


def probe_fps(video: Path) -> str:
    """Read the source frame rate so the overlay plays back at the original speed."""
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(video),
        ],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip() or "30"


def render_video(trace_path: Path, video: Path, frames: dict, bin_path) -> Path:
    """Burn masks into every traced frame and re-encode at the source frame rate."""
    fps = probe_fps(video)
    stem = trace_path.stem
    out_path = trace_path.with_name(f"{stem}_overlay.mp4")

    with tempfile.TemporaryDirectory() as tmp:
        src_dir, dst_dir = Path(tmp) / "src", Path(tmp) / "dst"
        src_dir.mkdir()
        dst_dir.mkdir()

        # Decode once, in order. -start_number 0 keeps ffmpeg's numbering aligned with the
        # trace's frame indices, which are 0-based.
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(video),
             "-start_number", "0", str(src_dir / "%06d.png")],
            check=True,
        )

        decoded = sorted(src_dir.glob("*.png"))
        for path in decoded:
            idx = int(path.stem)
            objects = frames.get(str(idx), {}).get("objects", [])
            panel = draw_panel(Image.open(path).convert("RGB"), objects, bin_path, idx)
            panel.save(dst_dir / path.name)

        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-framerate", fps,
             "-start_number", "0", "-i", str(dst_dir / "%06d.png"),
             # yuv420p for players that refuse anything else; pad guards odd dimensions.
             "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_path)],
            check=True,
        )

    traced = sum(1 for p in decoded if str(int(p.stem)) in frames)
    print(f"  {stem}: {traced}/{len(decoded)} frames traced @ {fps} fps -> {out_path}")
    return out_path


def contact_sheet(panels: list, cols: int = 3, pad: int = 6) -> "Image.Image":
    """Tile panels into a grid on a dark ground."""
    if not panels:
        raise ValueError("nothing to tile")
    w, h = panels[0].size
    rows = (len(panels) + cols - 1) // cols
    sheet = Image.new(
        "RGB", (cols * w + pad * (cols + 1), rows * h + pad * (rows + 1)), (18, 18, 20)
    )
    for i, panel in enumerate(panels):
        r, c = divmod(i, cols)
        sheet.paste(panel, (pad + c * (w + pad), pad + r * (h + pad)))
    return sheet


def visualize(
    trace_path: Path, clips_dir: Path, n_frames: int, cols: int, as_video: bool
) -> Path | None:
    trace = json.loads(trace_path.read_text())
    stem = trace_path.stem
    frames = trace["frames"]

    video = next((p for p in clips_dir.glob(f"{stem}.*") if p.suffix != ".json"), None)
    if video is None:
        print(f"  {stem}: no source clip found in {clips_dir}/, skipping", file=sys.stderr)
        return None

    bin_path = trace_path.with_suffix(".masks.bin")
    if not bin_path.exists():
        print(
            f"  {stem}: no {bin_path.name} — trace was written without --return-masks, "
            f"drawing boxes only"
        )
        bin_path = None

    if as_video:
        return render_video(trace_path, video, frames, bin_path)

    # Prefer frames that actually have detections; a sheet of empty frames teaches nothing.
    populated = [i for i in sorted(frames, key=int) if frames[i]["objects"]]
    pool = populated or sorted(frames, key=int)
    if not pool:
        print(f"  {stem}: trace has no frames, skipping", file=sys.stderr)
        return None

    step = max(1, len(pool) // n_frames)
    chosen = pool[::step][:n_frames]
    chosen_ints = [int(i) for i in chosen]

    with tempfile.TemporaryDirectory() as tmp:
        images = extract_frames(video, chosen_ints, Path(tmp))
        panels = [
            draw_panel(images[i], frames[str(i)]["objects"], bin_path, i)
            for i in chosen_ints
            if i in images
        ]

    if not panels:
        print(f"  {stem}: no frames decoded, skipping", file=sys.stderr)
        return None

    out_path = trace_path.with_name(f"{stem}_overlay.png")
    contact_sheet(panels, cols=cols).save(out_path)

    tracks = {o["object_id"] for f in frames.values() for o in f["objects"]}
    print(f"  {stem}: {len(panels)} panels, {len(tracks)} tracks -> {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render SAM 3 mask overlays from a trace.")
    parser.add_argument("--trace", help="Path to one sam3_out/<clip>.json")
    parser.add_argument("--all", action="store_true", help="Render every trace in --out-dir")
    parser.add_argument("--out-dir", default="sam3_out", help="Where traces live")
    parser.add_argument("--clips", default="clips", help="Where the source videos live")
    parser.add_argument("--frames", type=int, default=6, help="Panels per sheet")
    parser.add_argument("--cols", type=int, default=3, help="Grid columns")
    parser.add_argument(
        "--video", action="store_true", help="Render an overlay mp4 instead of a contact sheet"
    )
    args = parser.parse_args()

    clips_dir = Path(args.clips)
    if args.all:
        traces = sorted(
            p for p in Path(args.out_dir).glob("*.json") if not p.name.endswith(".masks.json")
        )
    elif args.trace:
        traces = [Path(args.trace)]
    else:
        parser.error("pass --trace <file> or --all")

    if not traces:
        sys.exit(f"no traces found in {args.out_dir}/")

    print(f"rendering {len(traces)} trace(s) as {'video' if args.video else 'contact sheet'}")
    for t in traces:
        visualize(t, clips_dir, args.frames, args.cols, args.video)


if __name__ == "__main__":
    main()
