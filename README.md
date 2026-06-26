# TVA Scoring

Patch-wise inference for scoring colorectal polyp tubulovillous architecture on whole-slide images. The model uses a CONCH 1.5 pre-trained backbone with a lightweight classification head and was trained on a dataset of annotated slides. The backbone is fine-tuned along with the classification head.

This repository is intended for running our trained TVA scoring model on unseen slides. It tiles a slide, scores each patch, and writes:

- a colour-mapped PNG score heatmap
- an optional CSV containing one row per scored patch

Pre-trained weights are provided here:
https://drive.google.com/drive/folders/1_4avGgOXCWblTGjaGr-KVDaMDDQ96dga?usp=sharing

+1 denotes a highly villous patch, -1 denotes a highly tubular patch, and 0 is neutral/intermediate. The model outputs are continuous in `[-1, 1]`.

## Setup

Use Python 3.11 or newer from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

`requirements.txt` installs TIAToolbox, PyTorch, OpenSlide support, CONCH, and the other packages imported by the inference code.

## Model Files

Download the pre-trained model files and keep the checkpoint together with its configuration files in the same directory. The scoring command expects this layout:

```text
path/to/model_dir/
  best.pt        # or another checkpoint .pt file
  train.yaml     # model architecture used to load the checkpoint
  dataset.yaml   # patch size, stride, resolution, and slide settings
```

Pass the checkpoint file itself to `tvs-score`, not the directory.

## Score One Slide

Run inference on a whole-slide image:

```bash
tvs-score /path/to/unseen_slide.ndpi /path/to/model_dir/best.pt \
  --output-dir scores \
  --save-csv
```

By default, inference uses an on-the-fly Otsu tissue mask, so background is skipped. For a slide named `unseen_slide.ndpi`, the command writes:

```text
scores/unseen_slide.png
scores/unseen_slide.csv
```

The PNG is a low-resolution heatmap at 6.0 microns-per-pixel by default. Blue corresponds to lower scores, red to higher scores. Model outputs are scalar patch scores in `[-1, 1]`, where the training labels map approximately to:

- `-1`: tubular
- `0`: neutral/intermediate
- `1`: villous

The CSV columns are:

```text
x,y,width,height,score
```

`x`, `y`, `width`, and `height` are patch bounds in level-0 slide pixel coordinates. `score` is the model prediction for that patch.

## Useful Options

Change the inference batch size:

```bash
tvs-score slide.ndpi model_dir/best.pt --batch-size 32 --save-csv
```

Change the output heatmap resolution:

```bash
tvs-score slide.ndpi model_dir/best.pt --heatmap-mpp 4.0 --save-csv
```

Score every tiled location, including background, without the Otsu restriction:

```bash
tvs-score slide.ndpi model_dir/best.pt --restrict-inference none --save-csv
```

Restrict inference to an annotation database:

```bash
tvs-score slide.ndpi model_dir/best.pt \
  --restrict-inference annotations \
  --annotation-db /path/to/slide.db \
  --output-dir scores \
  --save-csv
```

This will restrict inference to only regions covered by an annotation.

If you have QuPath GeoJSON annotations, convert them first:

```bash
tvs-prepare-annotations --source /path/to/geojson_dir --dest /path/to/db_dir
```

## Batch Scoring

You can also score a manifest CSV with `slide_id` and `path` columns. With the default tissue mask, the `path` value is not used but the column must exist:

```csv
slide_id,path
slide_001,
slide_002,
```

Create a data paths file pointing to the directory containing those slides:

```yaml
slides:
  - /path/to/slides
```

Then run:

```bash
tvs-score slides.csv /path/to/model_dir/best.pt \
  --data-paths data_paths.yaml \
  --output-dir scores \
  --save-csv
```

For manifest input, the slide file extension is read from the model directory's `dataset.yaml`.

## Troubleshooting

- `Missing train.yaml next to ...`: keep `train.yaml` in the same directory as the checkpoint.
- `No module named 'conch'`: install with `python -m pip install -r requirements.txt`.
- Slide loading errors usually mean OpenSlide/TIAToolbox cannot read the file or its format is unsupported in the current environment.
- If output is unexpectedly sparse, try `--restrict-inference none` to check whether tissue masking is excluding regions.
