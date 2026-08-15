"""
Pull a handful of short sample clips into ./clips for smoke-testing the SAM 3 app.

    python sam3/fetch_samples.py

These are generic test fixtures, not EgoVerse episodes — they exist to prove the Modal
plumbing and the text-prompt path work end to end before real zarr data is wired in.
"""

import shutil
import sys
import urllib.request
from pathlib import Path

CLIPS_DIR = Path("clips")

# Small public fixtures. tennis is the interesting one: person + racket + ball in the same
# clip, so a multi-concept prompt has something to actually discriminate.
REMOTE = {
    "bedroom.mp4": "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/bedroom.mp4",
    "tennis.mp4": "https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/tennis.mp4",
}

# skvideo ships sample clips and is already installed in the sibling EgoVerse env.
LOCAL_CANDIDATES = [
    Path.home() / "Github/EgoVerse/emimic/lib/python3.11/site-packages/skvideo/datasets/data/bikes.mp4",
    Path.home() / "Github/EgoVerse/emimic/lib/python3.11/site-packages/skvideo/datasets/data/carphone_pristine.mp4",
]


def main() -> None:
    CLIPS_DIR.mkdir(exist_ok=True)

    for name, url in REMOTE.items():
        dest = CLIPS_DIR / name
        if dest.exists():
            print(f"  {name} already present, skipping")
            continue
        print(f"  downloading {name} ...")
        urllib.request.urlretrieve(url, dest)

    for src in LOCAL_CANDIDATES:
        dest = CLIPS_DIR / src.name
        if dest.exists():
            print(f"  {src.name} already present, skipping")
        elif src.exists():
            shutil.copy(src, dest)
            print(f"  copied {src.name}")
        else:
            print(f"  (skipped {src.name} — not found at {src})", file=sys.stderr)

    clips = sorted(CLIPS_DIR.glob("*.mp4"))
    total = sum(p.stat().st_size for p in clips)
    print(f"\n{len(clips)} clips in {CLIPS_DIR}/ ({total / 1e6:.1f} MB):")
    for p in clips:
        print(f"  {p.name}  {p.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
