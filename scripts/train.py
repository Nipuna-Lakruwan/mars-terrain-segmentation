"""Train the U-Net on AI4Mars (256x256 Navcam images).

    uv run python scripts/train.py
    uv run python scripts/train.py --epochs 1 --limit 200   # smoke test

Writes checkpoints/unet.pt (best validation mIoU) and results/train_history.json.
The official expert-labelled test set is not touched here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from marsseg import CLASSES, IGNORE  # noqa: E402
from marsseg.metrics import confusion, summary  # noqa: E402
from marsseg.model import UNetResNet18  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
CKPT = ROOT / "checkpoints" / "unet.pt"
SEED, BATCH, LR = 42, 16, 3e-4
MEAN, STD = 0.449 * 255, 0.226 * 255  # ImageNet grey-level statistics (what the encoder expects)


def to_tensor(img: np.ndarray) -> torch.Tensor:
    return ((torch.from_numpy(img).float() - MEAN) / STD)[:, None]  # (B, 1, H, W)


def class_weights(labels: np.ndarray) -> torch.Tensor:
    """Rare classes (big rock is ~1% of pixels) get a bigger say in the loss.

    1/sqrt(frequency) is a gentle version of "inverse frequency": it helps rare classes
    without making the model paranoid about them.
    """
    counts = np.bincount(labels[labels != IGNORE].ravel(), minlength=len(CLASSES)).astype(float)
    w = 1 / np.sqrt(counts / counts.sum())
    return torch.tensor(w / w.mean(), dtype=torch.float32)


@torch.no_grad()
def evaluate(model, images, labels, device, batch: int = 32) -> dict:
    model.eval()
    cm = np.zeros((len(CLASSES), len(CLASSES)), np.int64)
    for i in range(0, len(images), batch):
        pred = model(to_tensor(images[i : i + batch]).to(device)).argmax(1).cpu().numpy()
        cm += confusion(pred, labels[i : i + batch])
    return summary(cm)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    tr, va = np.load(DATA / "train.npz"), np.load(DATA / "val.npz")
    Xtr, Ytr = tr["image"][: args.limit], tr["label"][: args.limit]
    Xva, Yva = va["image"], va["label"]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    w = class_weights(Ytr)
    print(f"train {len(Xtr)} | val {len(Xva)} | device {device} | class weights "
          + ", ".join(f"{c} {v:.2f}" for c, v in zip(CLASSES, w)))

    model = UNetResNet18(len(CLASSES)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    steps = args.epochs * ((len(Xtr) + BATCH - 1) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)
    loss_fn = nn.CrossEntropyLoss(weight=w.to(device), ignore_index=IGNORE)

    best, history, t0 = -1.0, [], time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm, total = rng.permutation(len(Xtr)), 0.0
        for i in range(0, len(perm), BATCH):
            b = np.sort(perm[i : i + BATCH])
            x, y = to_tensor(Xtr[b]), torch.from_numpy(Ytr[b]).long()
            # Augmentation: mirror left-right (terrain has no handedness) and jitter
            # brightness/contrast (sun angle and exposure vary a lot between sols).
            flip = torch.rand(len(b)) < 0.5
            x[flip], y[flip] = x[flip].flip(-1), y[flip].flip(-1)
            x = x * (1 + 0.2 * (torch.rand(len(b), 1, 1, 1) - 0.5)) + 0.2 * (torch.rand(len(b), 1, 1, 1) - 0.5)
            opt.zero_grad()
            loss = loss_fn(model(x.to(device)), y.to(device))
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(b)
        val = evaluate(model, Xva, Yva, device)
        history.append({"epoch": epoch, "train_loss": total / len(Xtr), "val_mIoU": val["mIoU"],
                        "val_iou": val["iou"], "minutes": round((time.time() - t0) / 60, 1)})
        print(f"epoch {epoch:2d}  loss {total / len(Xtr):.4f}  val mIoU {val['mIoU']:.4f}  "
              f"{val['iou']}  ({history[-1]['minutes']} min)", flush=True)
        if val["mIoU"] > best:
            best = val["mIoU"]
            CKPT.parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(), CKPT)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "train_history.json").write_text(json.dumps(history, indent=1))
    print(f"best val mIoU {best:.4f}")


if __name__ == "__main__":
    main()
