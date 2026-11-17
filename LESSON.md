# How it works: teaching a rover to read the ground

A lesson for a software engineer moving into AI engineering. Concepts first, then a walk
through the code in the order the data flows.

---

## Part 1: Concepts

### 1.1 Why Mars rovers care about terrain

Curiosity's wheels were badly damaged by sharp rocks in 2013, and Spirit died stuck in soft
sand in 2009. Rover drivers on Earth plan routes from the rover's **Navcam** images (black
and white, 1024×1024), and an onboard system that understands terrain could drive further
on its own between commands.

**AI4Mars** (NASA JPL, 2021) crowd-sourced labels for tens of thousands of Navcam images
through a Zooniverse project. Every pixel was labelled as one of:

| id | class | why it matters to a driver |
|---|---|---|
| 0 | soil | the default: drivable |
| 1 | bedrock | solid, but can be sharp |
| 2 | sand | the trap: wheels can sink |
| 3 | big rock | obstacle (larger than about 30 cm) |
| 255 | *no label* | unlabelled, the rover's own body, or terrain beyond 30 m |

The **test set** is special: 322 images labelled independently by several **expert** planetary
scientists. `min3` keeps only pixels where at least 3 experts agreed (a strict gold
standard); `min1` keeps any pixel an expert labelled (more coverage, more disagreement).

### 1.2 Classification vs segmentation

Image **classification** answers "what is in this picture?" with one label per image.
**Semantic segmentation** answers it for **every pixel**: the output is an image of class IDs
the same size as the input. That's what a path planner needs: *where* is the sand.

### 1.3 The U-Net shape

A segmentation network has two jobs that pull in opposite directions:

- **What**: recognising sand needs *context*, a wide view of texture and surroundings. Deep
  CNN layers have it, but at low resolution (our deepest features are 8×8).
- **Where**: drawing a crisp boundary needs *full resolution*, which only shallow layers have.

**U-Net** does both. An **encoder** shrinks the image step by step into rich, coarse features.
A **decoder** grows it back, and at each size it **concatenates** the encoder features of that
same size (the *skip connections*). The decoder ends up with "this region is sand" from deep
layers and "the edge is exactly here" from shallow layers.

```
256 ─────────────── skip ────────────────▶ 256  → 4 class scores per pixel
  128 ───────────── skip ──────────────▶ 128
     64 ─────────── skip ────────────▶ 64
        32 ──────── skip ─────────▶ 32
           16 ───── skip ──────▶ 16
               8  (encoder bottom: most context, least detail)
```

### 1.4 Transfer learning: don't start from zero

We have about 3,000 training images, which is small for a deep vision model. So the encoder is
a **ResNet-18 pretrained on ImageNet** (1.2 million everyday photos). Mars isn't a photo of
a cat, but early vision features (edges, textures, gradients, blobs) are universal. We
start with those weights and **fine-tune** everything on Mars. This is the most common
pattern in applied computer vision, and almost always beats training from scratch on small
data.

> Software analogy: depending on a well-tested library and patching it, instead of
> rewriting it.

### 1.5 Imbalance and the loss

Soil and bedrock cover most pixels; big rocks are a few percent. With plain cross-entropy
the model can score well by ignoring rocks. We **weight** each class's loss by
`1 / sqrt(frequency)`, which gently boosts the rare classes. Full inverse frequency would
make the model see rocks everywhere.

`ignore_index=255` tells the loss to skip unlabelled pixels completely. We never punish the
model for pixels nobody labelled.

### 1.6 Measuring honestly: IoU

**Pixel accuracy** is misleading under imbalance: call everything soil and you are often
right. The standard metric is **IoU (Intersection over Union)** per class:

```
IoU(sand) = pixels both we and experts call sand
            ────────────────────────────────────────────────
            pixels either we or experts (or both) call sand
```

It punishes both missing sand *and* inventing it. **mIoU** averages the four classes, so a
rare class counts as much as soil. We also report the "everything is soil" baseline: any
model worth having must clear it by a wide margin.

---

## Part 2: The code, in the order data flows

```
download_data.py → prepare_data.py → train.py (uses model.py, metrics.py) → evaluate.py
```

### 2.1 `scripts/download_data.py`

3 of the 13 training shards (0, 6, 12: spread out, since shards follow image order and
neighbouring images often show the same scene) plus the expert test sets. Pinned to one
commit of the Hugging Face mirror.

### 2.2 `scripts/prepare_data.py`: get data into a fast shape once

- Decodes JPEG/PNG bytes from Parquet with Pillow.
- **Resizes** 1024 → 256. Images use bilinear smoothing. Labels use **nearest-neighbour**,
  because averaging label 1 (bedrock) with label 3 (big rock) would give 2 (sand), a
  class that isn't there. Classic bug.
- Burns the **rover mask** and **>30 m range mask** into the label as 255 (ignore).
- Saves compact `uint8` NumPy arrays. Training then never touches JPEG decoding, so each
  epoch is pure GPU work.

### 2.3 `src/marsseg/model.py`

- `resnet18(weights=IMAGENET1K_V1)` downloads the pretrained encoder into `TORCH_HOME`.
- **Grayscale trick**: ResNet's first layer expects 3 colour channels. Navcam is grey, so
  we sum the first layer's RGB weights into a single channel. Mathematically identical to
  feeding the grey image three times, at a third of the cost.
- `Up` = upsample → concatenate skip → two 3×3 convs with BatchNorm. Four of them climb back
  from 1/32 to 1/2 resolution; a final interpolate restores 256×256.

### 2.4 `scripts/train.py`

- **Normalisation**: subtract ImageNet's grey mean and divide by its std, so inputs look the
  way the pretrained encoder expects.
- **Augmentation**: left-right flips (terrain has no handedness) and random
  brightness/contrast (sun angle and exposure vary between sols). Same random flip applied
  to image *and* label.
- **AdamW + OneCycleLR**: the learning rate warms up, then anneals towards zero, a reliable
  default for fine-tuning.
- Validation mIoU each epoch; keep the best checkpoint.

### 2.5 `src/marsseg/metrics.py`

Everything comes from one **confusion matrix** `cm[true, predicted]`, built with a single
`np.bincount` trick (`n * true + pred` turns each pixel pair into one integer).
IoU, precision and recall are row/column/diagonal arithmetic on it. `tests/` checks the
maths on a 2×2 image you can verify by hand.

### 2.6 `scripts/evaluate.py`

Scores both expert test versions, the all-soil baseline, a confusion-matrix plot (which
classes get mistaken for which), and a best/typical/worst gallery.

---

## Part 3: Try it yourself

1. Set `pretrained=False` in `train.py` and retrain. How much does ImageNet pretraining buy?
2. Remove the class weights. What happens to big-rock IoU and to soil IoU?
3. Train at 384×384 (edit `SIZE` in `prepare_data.py`). Do small rocks get better?
