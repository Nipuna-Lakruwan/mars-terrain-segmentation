"""Decode the AI4Mars Parquet files into small, fast NumPy arrays.

    uv run python scripts/prepare_data.py

For every labelled image:
  * shrink 1024x1024 -> 256x256 (images with smooth resampling, labels with nearest-neighbour,
    because averaging the label "1 = bedrock" with "3 = big rock" would invent "2 = sand")
  * mark pixels as IGNORE (255) where the rover's own body is in view or the terrain is more
    than 30 m away. AI4Mars ships these masks because labellers were told not to label there.

Train images are split 90/10 into train/val (val is only for early stopping).
Test = the official expert-labelled set, in two versions: min3 (>=3 experts agreed, the
strict gold standard) and min1 (>=1 expert labelled; more pixels, more disagreement).

Writes data/processed/{train,val,test_min3,test_min1}.npz (about 300 MB total).
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
SIZE, IGNORE, SEED = 256, 255, 42


def decode(row) -> tuple[np.ndarray, np.ndarray] | None:
    if not row["has_labels"] or row["label_mask"] is None:
        return None
    img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("L")
    lab = np.array(Image.open(io.BytesIO(row["label_mask"]["bytes"])))
    for key in ("rover_mask", "range_mask"):
        if row.get(key) is not None:
            lab[np.array(Image.open(io.BytesIO(row[key]["bytes"]))) > 0] = IGNORE
    img = np.array(img.resize((SIZE, SIZE), Image.Resampling.BILINEAR), np.uint8)
    lab = np.array(Image.fromarray(lab).resize((SIZE, SIZE), Image.Resampling.NEAREST), np.uint8)
    if (lab != IGNORE).mean() < 0.01:  # almost nothing labelled: no training signal
        return None
    return img, lab


def convert(files: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    imgs, labs, names = [], [], []
    for f in files:
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            for row in pf.read_row_group(rg).to_pylist():
                out = decode(row)
                if out is not None:
                    imgs.append(out[0])
                    labs.append(out[1])
                    names.append(row["image"]["path"])
        print(f"  {f.name}: {len(imgs)} images so far", flush=True)
    return np.stack(imgs), np.stack(labs), np.array(names)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("test_min3", "test_min1"):
        x, y, n = convert([RAW / f"{split}-00000-of-00001.parquet"])
        np.savez_compressed(OUT / f"{split}.npz", image=x, label=y, name=n)
        print(f"{split}: {len(x)} images")
    x, y, n = convert(sorted(RAW.glob("train-*.parquet")))
    test_names = set(np.load(OUT / "test_min3.npz")["name"])
    keep = np.array([name not in test_names for name in n])  # paranoia: no test image in train
    x, y, n = x[keep], y[keep], n[keep]
    perm = np.random.default_rng(SEED).permutation(len(x))
    n_val = len(x) // 10
    for split, idx in (("val", perm[:n_val]), ("train", perm[n_val:])):
        np.savez_compressed(OUT / f"{split}.npz", image=x[idx], label=y[idx], name=n[idx])
        print(f"{split}: {len(idx)} images")
    if not keep.all():
        print(f"dropped {(~keep).sum()} train images that also appear in test", file=sys.stderr)


if __name__ == "__main__":
    main()
