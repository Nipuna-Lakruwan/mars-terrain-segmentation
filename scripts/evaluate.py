"""Score the U-Net on the official AI4Mars expert test set and draw the README plots.

    uv run python scripts/evaluate.py

Reports per-class IoU, mIoU and pixel accuracy on:
    test_min3  pixels where at least 3 expert labellers agreed  (the strict gold standard)
    test_min1  pixels at least 1 expert labelled                (more pixels, noisier)
Baseline: "everything is soil", the most common class. Any useful model must beat it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from marsseg import CLASSES, IGNORE  # noqa: E402
from marsseg.metrics import confusion, summary  # noqa: E402
from marsseg.model import UNetResNet18  # noqa: E402
from train import to_tensor  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA, RESULTS, PLOTS = ROOT / "data" / "processed", ROOT / "results", ROOT / "results" / "plots"
COLORS = ["#c8874a", "#6b6b8a", "#f2d16b", "#b03a2e"]  # soil, bedrock, sand, big rock


@torch.no_grad()
def predict(model, images, device, batch: int = 16) -> np.ndarray:
    return np.concatenate([model(to_tensor(images[i : i + batch]).to(device)).argmax(1).cpu().numpy().astype(np.uint8)
                           for i in range(0, len(images), batch)])


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = UNetResNet18(len(CLASSES), pretrained=False).to(device)
    model.load_state_dict(torch.load(ROOT / "checkpoints" / "unet.pt", map_location=device))
    model.eval()

    metrics, cms = {}, {}
    for split in ("test_min3", "test_min1"):
        d = np.load(DATA / f"{split}.npz")
        pred = predict(model, d["image"], device)
        cm = confusion(pred, d["label"])
        cms[split] = cm
        soil = confusion(np.zeros_like(d["label"]), d["label"])  # predict class 0 everywhere
        metrics[split] = {"images": int(len(pred)), "unet": summary(cm), "all_soil_baseline": summary(soil),
                          "confusion_matrix_rows_true_cols_pred": cm.tolist()}
        if split == "test_min3":
            per_image = []
            for i in range(len(pred)):
                c = confusion(pred[i], d["label"][i])
                per_image.append(np.diag(c).sum() / max(c.sum(), 1))
            metrics[split]["per_image_accuracy_quartiles"] = [round(float(q), 3) for q in np.percentile(per_image, [25, 50, 75])]
            plot_examples(d, pred, np.array(per_image))
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    for split, m in metrics.items():
        u, b = m["unet"], m["all_soil_baseline"]
        print(f"{split}: U-Net mIoU {u['mIoU']:.3f}, pixel acc {u['pixel_accuracy']:.3f} | "
              f"all-soil mIoU {b['mIoU']:.3f}, acc {b['pixel_accuracy']:.3f}\n   IoU {u['iou']}")
    plot_confusion(cms["test_min3"])


def plot_confusion(cm: np.ndarray) -> None:
    norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "#222", fontsize=9)
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("model says")
    ax.set_ylabel("experts say")
    ax.set_title("Test (min3): share of each true class", fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOTS / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_examples(d, pred, acc) -> None:
    """Best, median and worst test images: Navcam image, expert labels, model output."""
    order = np.argsort(acc)
    picks = [("best", order[-1]), ("typical", order[len(order) // 2]), ("worst", order[0])]
    cmap = ListedColormap(COLORS)
    fig, axes = plt.subplots(3, 3, figsize=(9, 9.4))
    for row, (name, i) in enumerate(picks):
        lab = np.ma.masked_equal(d["label"][i], IGNORE)
        axes[row, 0].imshow(d["image"][i], cmap="gray")
        axes[row, 1].imshow(d["image"][i], cmap="gray")
        axes[row, 1].imshow(lab, cmap=cmap, vmin=0, vmax=3, alpha=0.6, interpolation="nearest")
        axes[row, 2].imshow(d["image"][i], cmap="gray")
        axes[row, 2].imshow(pred[i], cmap=cmap, vmin=0, vmax=3, alpha=0.6, interpolation="nearest")
        axes[row, 0].set_ylabel(f"{name}\n(pixel acc {acc[i]:.2f})", fontsize=9)
        for a in axes[row]:
            a.set_xticks([])
            a.set_yticks([])
    for a, t in zip(axes[0], ["Navcam image", "expert labels (min3)", "model prediction"]):
        a.set_title(t, fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS]
    fig.legend(handles, CLASSES, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(PLOTS / "examples.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
