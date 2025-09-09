# TubuloVillous Scoring (tvs)

A research project to score colorectal polyp architecture on whole slide images using TIAToolbox and PyTorch.

## Installation

```bash
pip install -e .
```

## Quick Start

1. **Prepare annotation stores**

   Convert QuPath-exported GeoJSON files to SQLite stores for fast access:

   ```bash
   tvs-prepare-annotations --data-paths configs/data_paths.yaml
   ```

2. **Create train/val/test splits**

   ```bash
   tvs-split path/to/annotations/ splits/ --val 0.2 --test 0.2
   ```

3. **Train a model**

   ```bash
   tvs-train configs/train_cnn.yaml \
     --dataset-config configs/dataset.yaml \
     --data-paths configs/data_paths.yaml \
     --train-manifest splits/train.csv \
     --val-manifest splits/val.csv
   ```

   The dataset configuration includes `slide_extension` to specify the slide file
   extension (default: `svs`).

4. **Evaluate predictions**

   Provide a CSV with `prediction` and `target` columns to compute metrics,
   a reliability plot, and a ROC curve (T vs V only):

   ```bash
   tvs-evaluate predictions.csv
   ```

5. **Score a whole-slide image**

   ```bash
   tvs-score slide.ext model.ts --annotation-db path/to/annotations/slide.db
   ```

   The command saves a heatmap PNG (`score.png`) at the specified resolution.

## Development

Install dependencies and enable pre-commit hooks:

```bash
pip install -r requirements.txt
pre-commit install
```

## Testing

Run the unit tests with:

```bash
pytest
```
