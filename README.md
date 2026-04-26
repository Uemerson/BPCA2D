# BPCA2D

2D Block-Based PCA Pooling Layer — a custom spatial pooling layer that replaces MaxPooling by projecting local patches onto their first principal component.

All methods train a ResNet-style CNN on the [ImageNet64](https://patrykchrabaszcz.github.io/Imagenet32/) dataset (64×64, 1000 classes). The network architecture is shared across all variants: 4 convolutional blocks (32→64→128→256 channels) that progressively downscale from 64×64 to 4×4, followed by a fully-connected classifier (4096→1024→512→1000).

---

## BPCA2D Layer

The `BPCA2D` module is a drop-in replacement for `MaxPool2d`. For a given `kernel_size` and `stride`, it:

1. Extracts non-overlapping patches from the input feature map using `torch.nn.functional.unfold`.
2. Centers the patches and computes PCA to find the dominant direction of variance.
3. Projects each patch onto the first principal component, producing a single scalar per patch — equivalent to a learned, data-driven pooling.

Parameters: `kernel_size`, `stride`, `q` (number of principal components kept; default `1`).

---

## Methods

### `resnet.py` — Baseline ResNet

Standard ResNet-style baseline. All 4 blocks use `MaxPool2d(2, 2)` for downsampling. No BPCA2D.

| Block | Pooling     |
|-------|-------------|
| 1–4   | MaxPool2d   |

---

### `bpca_1.py` — BPCA2D with `pca_lowrank` (per-image)

Replaces all `MaxPool2d` layers with `BPCA2D`. PCA is computed **per image** in a Python loop using `torch.pca_lowrank`. Each image gets its own principal component computed from all its patches across all channels.

| Block | Pooling      |
|-------|--------------|
| 1–4   | BPCA2D       |

**PCA method:** `torch.pca_lowrank` · **Scope:** per-image (loop)

---

### `bpca_2.py` — BPCA2D with SVD (batched)

Replaces all `MaxPool2d` layers with `BPCA2D`. PCA is computed **across the entire batch** at once using `torch.linalg.svd` on the centered `[B, C·L, K·K]` patch matrix. Fully vectorized — no Python loop over images.

| Block | Pooling      |
|-------|--------------|
| 1–4   | BPCA2D       |

**PCA method:** `torch.linalg.svd` · **Scope:** batch-level (vectorized)

---

### `bpca_3.py` — BPCA2D with covariance + `eigh` (batched)

Replaces all `MaxPool2d` layers with `BPCA2D`. PCA is computed **across the entire batch** by forming the covariance matrix `XᵀX` and extracting its largest eigenvector via `torch.linalg.eigh`. Fully vectorized.

| Block | Pooling      |
|-------|--------------|
| 1–4   | BPCA2D       |

**PCA method:** covariance + `torch.linalg.eigh` · **Scope:** batch-level (vectorized)

---

### `bpca_4.py` — BPCA2D with covariance + `eigh` (per-image)

Replaces all `MaxPool2d` layers with `BPCA2D`. Same covariance + `eigh` approach as `bpca_3`, but PCA is computed **per image** in a Python loop, giving each image its own principal component.

| Block | Pooling      |
|-------|--------------|
| 1–4   | BPCA2D       |

**PCA method:** covariance + `torch.linalg.eigh` · **Scope:** per-image (loop)

---

### `resnet_bpca_3.py` — Hybrid: MaxPool in early blocks, BPCA2D in later blocks

BPCA2D (covariance + `eigh`, batched) is introduced only in the deeper blocks where feature maps are smaller, while early blocks retain `MaxPool2d`.

| Block | Pooling      |
|-------|--------------|
| 1–2   | MaxPool2d    |
| 3–4   | BPCA2D       |

---

### `resnet_bpca_4.py` — Hybrid: MaxPool in first block, BPCA2D in remaining blocks

Extends the BPCA2D coverage compared to `resnet_bpca_3`: only the first block uses `MaxPool2d`, and all subsequent blocks use BPCA2D (covariance + `eigh`, batched).

| Block | Pooling      |
|-------|--------------|
| 1     | MaxPool2d    |
| 2–4   | BPCA2D       |

---

## Summary

| File               | BPCA2D scope    | PCA method              | Blocks with BPCA2D |
|--------------------|-----------------|-------------------------|--------------------|
| `resnet.py`        | —               | —                       | None               |
| `bpca_1.py`        | per-image       | `pca_lowrank`           | 1, 2, 3, 4         |
| `bpca_2.py`        | batch-level     | `linalg.svd`            | 1, 2, 3, 4         |
| `bpca_3.py`        | batch-level     | covariance + `eigh`     | 1, 2, 3, 4         |
| `bpca_4.py`        | per-image       | covariance + `eigh`     | 1, 2, 3, 4         |
| `resnet_bpca_3.py` | batch-level     | covariance + `eigh`     | 3, 4               |
| `resnet_bpca_4.py` | batch-level     | covariance + `eigh`     | 2, 3, 4            |

