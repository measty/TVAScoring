# TubuloVillous — Colorectal Polyp Architecture Scorer

Train a model to assign a continuous score to WSI patches indicating where architecture is *tubular* (−1) versus *villous* (+1), using a small set of WSIs with QuPath annotations. The project uses **TIAToolbox** for slide/annotation I/O and **SlidingWindowPatchExtractor** to sample patches restricted to annotation-defined regions via an AnnotationStore mask + filter string.

## Functionality

* Convert QuPath `.geojson` files → TIAToolbox `SQLiteStore` (.db). Build with `SQLiteStore.from_geojson(...)` then persist via `store.dump(db_path)` for fast re‑use.
* Read WSIs via TIAToolbox (`WSIReader`) at a chosen resolution (e.g., 0.5 µm/px).
* Build a grouped **train/val/test** split at the WSI level.
* Wrap **SlidingWindowPatchExtractor** for class-filtered patch sampling using `store_filter` (e.g., `props['class'] == 'villous'`).
* PyTorch `Dataset`/`DataLoader` yielding RGB patches and a regression target in **\[−1, 1]** (villous=+1, tubular=−1, optional tubulovillous=0).
* Two model options:

  * **CNN baseline** trained from scratch.
  * **Transformer/ViT head on top of Conch** patch embeddings (Mahmood Lab), w/ optional freezing of the backbone.
* Training engine with mixed precision, early stopping, and detailed logging (tqdm + rich logging).
* Evaluation: RMSE/MAE + Spearman/Pearson against target scores; class-wise calibration plots.
* Inference: whole-slide dense scoring (tiled) with stitching to produce a heatmap/scoremap TIFF/NPY; optional on-annotation inference.
* Visualization utilities to overlay diverging colormap scoremaps on WSIs; quick-look scripts to preview sampled patches per class.
* Reproducible configs, CLI entry points, and unit tests (pytest) for core components.

## Notes about Implementation

### Data & Annotations

* **Annotation ingestion**: Use `SQLiteStore.from_geojson(path)` and persist to a `.db` via `store.dump(db_path)` for fast subsequent loads.
* **Filter strings**: Leverage `SlidingWindowPatchExtractor(..., input_mask=annotation_store, store_filter="props['class'] == 'villous']")`-style filters. Provide a small helper to generate safe filters from a mapping `{label: score}`.
* **Label mapping**: default `{ 'tubular': -1.0, 'villous': +1.0, 'tubulovillous': 0.0 }` (last optional). Allow user-config overrides.
* **Min mask ratio**: set `min_mask_ratio` (e.g., 0.5) to ensure patches are mostly within annotation polygons.
* **Resolution/units**: prefer explicit physical units (`resolution: 0.5, units: "mpp"`) for consistent patch contents across slides.

### Dataset & Sampling

* Maintain **two (or more) extractors**, one per class-filter, merged by a balanced sampler to prevent label skew.
* Use **StrongAugment** ([https://github.com/jopo666/StrongAugment](https://github.com/jopo666/StrongAugment)) for augmentations within a torchvision pipeline; keep any additional ops (e.g., flips/resized crops/normalization) minimal and reproducible.
* Ensure **grouped splits** by slide (and patient when available) to avoid leakage.

### Modeling

* **Output range**: final layer uses `tanh` to enforce \[−1, 1]. Train with **Huber** or **MSE** loss.
* **CNN baseline**: compact ConvNet (or `timm` backbone) with global pooling + MLP head.
* **Conch-based model (ViT backbone)**: start from Mahmood Lab's Conch ViT patch-embedding model; attach a lightweight head (**linear** or **MLP**) *or* optionally replace the last transformer block with a task-specific variant. Fine‑tuning modes: **frozen backbone** (head‑only) or **end‑to‑end**; configure LR multipliers and weight decay accordingly.
* Log per-class residuals, reliability curves (bin score vs target), and patch montages for quick qualitative checks.

### Whole‑Slide Inference

* Use tissue mask (`input_mask='otsu'`/`'morphological'`) or on-annotation inference. Tiled sliding window with overlap; save as either an annotation store of patch scores (Polygon.from_bounds() for the geometry, score in props) or stitch via weighted averaging into a low-res heatmap & save as .png.

### Performance & Reproducibility

* Mixed precision (autocast+GradScaler), deterministic seeds, and cudnn benchmarking guarded.
* Extensive logging via `rich` + `tqdm`; checkpoints with best-val metric and last.
* Unit tests include tiny synthetic WSI + toy annotations to validate geometry, filtering, and dataset sampling.

## Additional Context

* **Annotation property names** from QuPath must match filters (commonly `props['class']`). Expose a config to customize the property key if users use a different field (e.g., `annotation_type`).
* **Coordinate pitfalls**: Always compute scale factors from baseline MPP if working at non-baseline mpp and scale coordinates appropriately; never assume pyramid levels map 1:1 across scanners. 
* **Patch borders**: set `within_bound=False` to allow padding so border regions are included; prefer `pad_mode='constant'` with black padding.
* **Small‑data regime**: favor strong augmentation and moderate regularization; consider freezing Conch for initial runs.

## General Instructions (Important!)

1. Code should generally be written in python. Any appriopriate, well-maintained libraries may be used (add new dependencies into requirements.txt). HTML/javascript or similar could be used for interactive visualization purposes. Pytorch should be used for any ML needs.
2. Code should be well documented, with docstrings and comments where appropriate - but do not over-do comments on obvious code.
3. Users should be kept informed on what is happening while code is running via well-chosen logging at key points describing the task currently being performed (and on what, if appropriate), and tqdm should be used to provide progress indicators for potentially long-running loops.
4. Code should be well structured, with a clear directory structure and modular code.
5. Code should be well tested, with unit tests (use pytest) for all major functions and classes.
6. The main tools in the toolkit should be runnable from the command line (use Argparse).
7. When considering compute requirements, assume availability of a reasonably powerful modern workstation with a single 24Gb GPU and 128Gb ram. However, write code with efficiency and speed in mind, where possible.
8. When you feel it would be useful for organisational/planning purposes to modify or add notes/tasks/etc into this markdown file, you should do so. However, be concise. This will help you keep track of what you are doing, what you need to do, and how the repo is organised.
9. Complete a single task at a time, though it's good to keep other tasks in mind when deciding how to implement the current one.

## Proposed Directory Structure (Update if needed)

```text
tubulovillous/
├── README.md
├── pyproject.toml                    # Packaging & CLI entry points
├── AGENTS.md                         # This file
├── requirements.txt                  # tiatoolbox, torch, torchvision, strong-augment, numpy, pandas, rich, tqdm, shapely, pytest, opencv-python-headless, tifffile, pillow, pydantic, timm (optional), scikit-image (optional: albumentations)
├── .gitignore
├── .pre-commit-config.yaml
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── configs/
│   ├── data_paths.yaml               # roots for slides/annotations/cache/outputs
│   ├── dataset.yaml                  # patch_size, stride, resolution, units, min_mask_ratio, class→score map
│   ├── train_cnn.yaml                # hparams for CNN baseline
│   └── train_conch.yaml              # hparams for Conch head (weights path; head_type: linear|mlp|replace_last_block; finetune: frozen|end_to_end)
│
├── data/                             # (placeholder; not committed)
│   ├── raw/
│   └── processed/
│
├── scripts/                          # Thin wrappers around src.tvs.cli
│   ├── prepare_annotations.py        # geojson→SQLiteStore (.db) cache
│   ├── split_dataset.py              # train/val/test split by WSI/patient
│   ├── preview_patches.py            # quick visual sanity checks
│   ├── train.py                      # train selected model
│   ├── evaluate.py                   # eval on validation/test
│   └── score_wsi.py                  # whole-slide inference + overlays
│
├── notebooks/
│   ├── 01_data_overview.ipynb
│   └── 02_model_prototyping.ipynb
│
├── src/
│   └── tvs/                          # (tvs = TubuloVillouS)
│       ├── __init__.py
│       ├── io/
│       │   ├── __init__.py
│       │   └── annotations.py        # from_geojson → SQLiteStore (.db), cache, filter helpers
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── extractors.py         # wrappers for SlidingWindowPatchExtractor
│       │   ├── dataset.py            # PyTorch Dataset using TIAToolbox extractors
│       │   ├── samplers.py           # balanced & grouped samplers
│       │   └── transforms.py         # torchvision + StrongAugment pipelines
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── cnn_baseline.py       # small ConvNet/ResNet-ish
│       │   ├── conch_head.py         # Conch (ViT); linear/MLP head or replace last block → score
│       │   └── factory.py            # make_model(cfg)
│       │
│       ├── training/
│       │   ├── __init__.py
│       │   ├── engine.py             # train/validate loops (AMP, logging)
│       │   ├── losses.py             # Huber/MSE
│       │   └── metrics.py            # MAE/RMSE, Spearman/Pearson
│       │
│       ├── inference/
│       │   ├── __init__.py
│       │   ├── tiler.py              # sliding window over WSI -> save patch scores in annotation store
│       │   └── stitcher.py           # blend overlapping patches -> low-res scoremap .png
│       │
│       ├── viz/
│       │   ├── __init__.py
│       │   ├── overlays.py           # scoremap→RGB overlays on slide thumb, legends
│       │   └── top_examples.py       # montages of top-N patches per class
│       │
│       └── cli/
│           ├── __init__.py
│           ├── prepare_annotations.py
│           ├── split.py
│           ├── preview.py
│           ├── train.py
│           ├── evaluate.py
│           └── score.py
│
└── tests/
    ├── __init__.py
    ├── conftest.py                   # fixtures: tiny WSI, toy annotations
    ├── test_annotations.py
    ├── test_extractors.py
    ├── test_dataset.py
    ├── test_models.py
    ├── test_training.py
    └── test_inference.py
```

### 🗂️ Implementation Task Roadmap

| #      | Task                                      | Key Files / Paths                                                                                                                            | Core Functionality & Notes                                                                                                                  |
| ------ | ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| **0**  | **Bootstrap repo & packaging**            | `pyproject.toml`, `requirements.txt`, `.pre-commit-config.yaml`, `.github/workflows/ci.yml`, `README.md`                                     | Declare package `tvs`; set console entry points; pin deps (tiatoolbox, torch, albumentations, rich, tqdm, tifffile, shapely, timm, pytest). |
| **1**  | **Configs**                               | `configs/*.yaml`                                                                                                                             | Data roots, class→score mapping, patch size/stride, resolution/units, augmentation, training hparams.                                       |
| **2**  | **Annotation ingestion**                  | `src/tvs/io/annotations.py`, `scripts/prepare_annotations.py`, `src/tvs/cli/prepare_annotations.py`                                          | Load `.geojson` → `SQLiteStore` (TIAToolbox), validate class props, cache, simple stats per slide.                                      |
| **3**  | **Train/val/test split**                  | `scripts/split_dataset.py`, `src/tvs/cli/split.py`                                                                                           | Grouped split by WSI; write CSVs (split manifests).                                                                                 |
| **4**  | **Extractor wrappers**                    | `src/tvs/data/extractors.py`                                                                                                                 | Convenience around `SlidingWindowPatchExtractor` with `store_filter` per class, `min_mask_ratio`, padding, stride.                          |
| **5**  | **Dataset & Samplers**                    | `src/tvs/data/dataset.py`, `src/tvs/data/samplers.py`, `src/tvs/data/transforms.py`                                                          | Merge multiple class-conditioned extractors; balanced sampling; Albumentations pipelines.                                                   |
| **6**  | **Unit tests: IO & Dataset**              | `tests/test_annotations.py`, `tests/test_extractors.py`, `tests/test_dataset.py`, `tests/conftest.py`                                        | Synthetic WSI + polygons; verify patch coverage, label mapping, and deterministic splits.                                                   |
| **7**  | **CNN baseline**                          | `src/tvs/models/cnn_baseline.py`, `src/tvs/models/factory.py`                                                                                | Small ConvNet with global pooling → MLP → `tanh`.                                                                                           |
| **8**  | **Conch head**                            | `src/tvs/models/conch_head.py`, `src/tvs/models/factory.py`                                                                                  | Conch backbone; add transformer/MLP head; config for freeze vs fine-tune and weight init from checkpoint.                            |
| **9** | **Training engine & CLI**                 | `src/tvs/training/engine.py`, `src/tvs/training/losses.py`, `src/tvs/training/metrics.py`, `scripts/train.py`, `src/tvs/cli/train.py`        | AMP, grad clipping, early stopping, LR scheduling; rich+tqdm logging; checkpointing.                                                        |
| **10** | **Evaluation & preview tools**            | `scripts/evaluate.py`, `src/tvs/cli/evaluate.py`, `scripts/preview_patches.py`, `src/tvs/cli/preview.py`, `src/tvs/viz/overlays.py`          | Metric computation; reliability plots; patch montages; score overlays.                                                                      |
| **11** | **Whole‑slide inference**                 | `src/tvs/inference/tiler.py`, `src/tvs/inference/stitcher.py`, `src/tvs/inference/scorer.py`, `scripts/score_wsi.py`, `src/tvs/cli/score.py` | Dense scoring on tissue/annotations; write annotations store or stitch -> low res PNG overlays.                                                                 |
| **12** | **Unit tests: models/training/inference** | `tests/test_models.py`, `tests/test_training.py`, `tests/test_inference.py`                                                                  | Forward pass shapes; overfit-a-batch test; tiling/stitching correctness.                                                                    |
| **13** | **Docs & Cleanup**                        | `README.md`, `AGENTS.md`                                                                                                                     | Add quick-start, examples, finalize notes.                                                                                                  |

> **Tip:** After each task, add a brief ✓ entry under **AGENTS.md** so progress is explicit and new subtasks can be appended without losing context.

### Progress

* [x] 0 Bootstrap repo & packaging
* [x] 1 Configs — add data_paths, dataset, and training YAML templates
* [x] 2 Annotation ingestion — convert GeoJSON annotations to cached SQLite stores
* [x] 3 Train/val/test split
* [x] 4 Extractor wrappers — class-filtered patch extractors
* [ ] 5 Dataset & Samplers
* [ ] 6 Unit tests: IO & Dataset
  * [x] add annotation ingestion tests
* [ ] 7 CNN baseline
* [ ] 8 Conch head
* [ ] 9 Training engine & CLI
* [ ] 10 Evaluation & preview tools
* [ ] 11 Whole‑slide inference
* [ ] 12 Unit tests: models/training/inference
* [ ] 13 Docs & Cleanup
