"""Download a documented subset of NASA JPL's AI4Mars dataset (Curiosity / MSL imagery).

AI4Mars (Swan et al., CVPR Workshops 2021) is about 30 GB in its original form (MSL + MER
images and several label versions). We use the MSL part as re-packaged to Parquet on
Hugging Face (`hassanjbara/AI4MARS`, a copy of the Kaggle mirror), pinned to one commit:

    train      : 3 of the 13 shards (0, 6, 12) -> about 4,200 of 18,130 labelled images
    test_min3  : all 322 test images, labels where >= 3 expert annotators agreed (gold)
    test_min1  : the same 322 images, labels where >= 1 expert annotated (more pixels, noisier)

Total download: about 1.6 GB. Every training image in the official split comes from
crowd-sourced labels; the test images are the official expert-labelled ones, so the test
set is genuinely held out.

    uv run python scripts/download_data.py               # default subset
    uv run python scripts/download_data.py --train-shards 0 1 2 3 4 5 6 7 8 9 10 11 12  # all
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REPO = "hassanjbara/AI4MARS"
REVISION = "93dbb79de6cd234799750b6129fbfb3b197582fa"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/data"


def download(name: str) -> None:
    dest = RAW / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(f"{BASE}/{name}", headers={"User-Agent": "nasa-ml-download/0.1"})
    with urllib.request.urlopen(req, timeout=120) as r:
        total = int(r.headers.get("Content-Length", 0))
        if dest.exists() and dest.stat().st_size == total:
            print(f"  {name}: cached")
            return
        tmp, done = dest.with_suffix(".part"), 0
        with tmp.open("wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if done % (100 << 20) < (1 << 20):
                    print(f"  {name}: {done / 1e6:.0f}/{total / 1e6:.0f} MB", flush=True)
    tmp.rename(dest)
    print(f"  {name}: done", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-shards", type=int, nargs="*", default=[0, 6, 12])
    args = ap.parse_args()
    for split in ("test_min3", "test_min1"):
        download(f"{split}-00000-of-00001.parquet")
    for i in args.train_shards:
        download(f"train-{i:05d}-of-00013.parquet")


if __name__ == "__main__":
    main()
