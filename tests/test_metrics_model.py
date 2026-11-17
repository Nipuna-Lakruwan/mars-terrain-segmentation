import numpy as np
import torch

from marsseg import IGNORE
from marsseg.metrics import confusion, iou_per_class, summary
from marsseg.model import UNetResNet18


def test_ignored_pixels_never_count():
    label = np.array([[0, 1], [IGNORE, 3]])
    pred = np.array([[0, 2], [3, 3]])
    cm = confusion(pred, label)
    assert cm.sum() == 3
    assert cm[0, 0] == 1 and cm[1, 2] == 1 and cm[3, 3] == 1


def test_iou_by_hand():
    # class 0: 2 right, 1 predicted as 1; class 1: 1 right
    cm = np.array([[2, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    iou = iou_per_class(cm)
    assert np.isclose(iou[0], 2 / 3) and np.isclose(iou[1], 1 / 2)
    assert np.isnan(iou[2])  # class never present nor predicted: excluded from the mean
    assert np.isclose(summary(cm)["mIoU"], (2 / 3 + 1 / 2) / 2, atol=1e-4)


def test_unet_output_shape():
    model = UNetResNet18(4, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 1, 256, 256))
    assert out.shape == (2, 4, 256, 256)
