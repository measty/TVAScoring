"""Extract sampled TVA patches from NDPI slides and build a montage.

Example:
    python extract_tva_patches.py ^
        --samples-csv representative_tva_patches.csv ^
        --slide-root E:\\path\\to\\slides ^
        --output-dir extracted_tva_patches

The slide root must contain the same subfolder structure as the score CSVs,
with slide filenames matching the CSV stems and a .ndpi extension.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_LOCAL_TIATOOLBOX_ROOT = Path(r"C:\Users\meast\tiatoolbox")


@dataclass(frozen=True)
class PatchSample:
    sample_id: str
    label: str
    score: float
    x: float
    y: float
    width: float
    height: float
    slide_relative_path: Path

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        left = int(round(self.x))
        top = int(round(self.y))
        right = int(round(self.x + self.width))
        bottom = int(round(self.y + self.height))
        return left, top, right, bottom


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract TVA patch PNGs from NDPI slides using tiatoolbox WSIReader."
    )
    parser.add_argument(
        "--samples-csv",
        type=Path,
        required=True,
        help="CSV containing sampled patches, including x/y/width/height/score and slide_relative_path.",
    )
    parser.add_argument(
        "--slide-root",
        type=Path,
        required=True,
        help="Root folder containing NDPI slides in the same subfolder structure as the score CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("extracted_tva_patches"),
        help="Output directory for patch PNGs, manifest, and montage.",
    )
    parser.add_argument(
        "--read-resolution",
        type=float,
        default=0,
        help="Resolution argument passed to WSIReader.read_bounds. Default reads level 0.",
    )
    parser.add_argument(
        "--read-units",
        default="level",
        choices=["level", "baseline", "mpp", "power"],
        help="Units argument passed to WSIReader.read_bounds.",
    )
    parser.add_argument(
        "--thumbnail-size",
        type=int,
        default=256,
        help="Maximum width/height for each tile in the montage.",
    )
    parser.add_argument(
        "--montage-per-class",
        type=int,
        default=10,
        help="Number of strongest patches per class to include in the montage. Use 0 to show all.",
    )
    parser.add_argument(
        "--montage-columns",
        type=int,
        default=5,
        help="Number of columns in the montage figure.",
    )
    parser.add_argument(
        "--patch-max-size",
        type=int,
        default=0,
        help="If positive, resize saved patch PNGs so their longest edge is this size.",
    )
    parser.add_argument(
        "--montage-name",
        default="tva_patch_compilation.png",
        help="Filename for the montage PNG inside output-dir.",
    )
    parser.add_argument(
        "--manifest-name",
        default="extracted_patches_manifest.csv",
        help="Filename for the extraction manifest inside output-dir.",
    )
    parser.add_argument(
        "--tiatoolbox-root",
        type=Path,
        default=DEFAULT_LOCAL_TIATOOLBOX_ROOT,
        help="Optional local tiatoolbox source checkout to add to sys.path before import.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate CSV rows and slide paths without reading or writing patch images.",
    )
    return parser.parse_args()


def configure_tiatoolbox_path(tiatoolbox_root: Path) -> None:
    if tiatoolbox_root and tiatoolbox_root.exists():
        sys.path.insert(0, str(tiatoolbox_root))


def load_samples(samples_csv: Path) -> list[PatchSample]:
    samples: list[PatchSample] = []
    with samples_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"x", "y", "width", "height", "score"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required column(s) in {samples_csv}: {sorted(missing)}")

        for row_index, row in enumerate(reader, start=1):
            sample_id = row.get("sample_id") or f"sample_{row_index:03d}"
            label = row.get("class") or row.get("label") or "patch"
            slide_relative = row.get("slide_relative_path")
            if not slide_relative:
                slide_relative = derive_slide_relative_path(row, row_index)
            samples.append(
                PatchSample(
                    sample_id=sample_id,
                    label=label,
                    score=float(row["score"]),
                    x=float(row["x"]),
                    y=float(row["y"]),
                    width=float(row["width"]),
                    height=float(row["height"]),
                    slide_relative_path=Path(slide_relative),
                )
            )
    if not samples:
        raise ValueError(f"No rows found in {samples_csv}")
    return samples


def derive_slide_relative_path(row: dict[str, str], row_index: int) -> str:
    csv_relative = row.get("csv_relative_path")
    if csv_relative:
        csv_path = Path(csv_relative)
        return str(csv_path.with_suffix(".ndpi"))
    slide_subdir = row.get("slide_subdir")
    slide_stem = row.get("slide_stem")
    if slide_subdir is not None and slide_stem:
        return str(Path(slide_subdir) / f"{slide_stem}.ndpi")
    raise ValueError(
        "Row "
        f"{row_index} has no slide_relative_path and lacks csv_relative_path or slide_subdir/slide_stem."
    )


def sanitize_filename(value: str) -> str:
    value = value.strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def to_rgb_image(array: np.ndarray) -> Image.Image:
    arr = np.asarray(array)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            max_value = 1.0 if float(np.nanmax(arr)) <= 1.0 else 255.0
            arr = np.clip(arr, 0, max_value) / max_value * 255
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def resize_longest_edge(image: Image.Image, max_size: int) -> Image.Image:
    if max_size <= 0:
        return image
    width, height = image.size
    longest = max(width, height)
    if longest <= max_size:
        return image
    scale = max_size / longest
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def extract_patches(
    samples: Iterable[PatchSample],
    slide_root: Path,
    output_dir: Path,
    read_resolution: float,
    read_units: str,
    patch_max_size: int,
    dry_run: bool,
) -> list[dict[str, str]]:
    samples = list(samples)
    if dry_run:
        wsi_reader_class = None
    else:
        from tiatoolbox.wsicore.wsireader import WSIReader

        wsi_reader_class = WSIReader

    patch_dir = output_dir / "patches"
    if not dry_run:
        patch_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    index_width = max(2, len(str(len(samples))))

    for index, sample in enumerate(samples, start=1):
        slide_path = slide_root / sample.slide_relative_path
        if not slide_path.exists():
            raise FileNotFoundError(f"Slide not found for {sample.sample_id}: {slide_path}")

        safe_id = sanitize_filename(sample.sample_id)
        safe_label = sanitize_filename(sample.label)
        patch_name = f"{index:0{index_width}d}_{safe_label}_{safe_id}_score_{sample.score:+.6f}.png"
        patch_path = patch_dir / patch_name
        bounds = sample.bounds

        if dry_run:
            width = ""
            height = ""
        else:
            reader = wsi_reader_class.open(slide_path)
            patch_array = reader.read_bounds(
                bounds,
                resolution=read_resolution,
                units=read_units,
                coord_space="baseline",
            )
            image = to_rgb_image(patch_array)
            image = resize_longest_edge(image, patch_max_size)
            image.save(patch_path)
            width, height = image.size

        manifest_rows.append(
            {
                "sample_id": sample.sample_id,
                "class": sample.label,
                "score": f"{sample.score:.9f}",
                "slide_relative_path": sample.slide_relative_path.as_posix(),
                "slide_path": str(slide_path),
                "bounds_left": str(bounds[0]),
                "bounds_top": str(bounds[1]),
                "bounds_right": str(bounds[2]),
                "bounds_bottom": str(bounds[3]),
                "patch_png": "" if dry_run else str(patch_path),
                "patch_width": str(width),
                "patch_height": str(height),
            }
        )

    return manifest_rows


def write_manifest(rows: list[dict[str, str]], manifest_path: Path) -> None:
    if not rows:
        return
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def make_montage(
    manifest_rows: list[dict[str, str]],
    output_path: Path,
    thumbnail_size: int,
    montage_per_class: int,
    columns: int = 5,
) -> None:
    by_label: dict[str, list[dict[str, str]]] = {}
    for row in manifest_rows:
        by_label.setdefault(row["class"], []).append(row)

    for label, label_rows in list(by_label.items()):
        label_lower = label.lower()
        if label_lower == "tubular":
            sorted_rows = sorted(label_rows, key=lambda row: float(row["score"]))
        else:
            sorted_rows = sorted(label_rows, key=lambda row: float(row["score"]), reverse=True)
        if montage_per_class > 0:
            sorted_rows = sorted_rows[:montage_per_class]
        by_label[label] = sorted_rows

    label_order = sorted(
        by_label,
        key=lambda label: (0 if label.lower() == "villous" else 1 if label.lower() == "tubular" else 2, label),
    )
    title_font = load_font(24, bold=True)
    label_font = load_font(18, bold=True)
    text_font = load_font(13)
    small_font = load_font(12)

    margin = 28
    gap = 14
    header_h = 42
    group_label_h = 30
    caption_h = 42
    tile_w = thumbnail_size
    tile_h = thumbnail_size + caption_h

    group_heights = []
    for label in label_order:
        rows_needed = int(np.ceil(len(by_label[label]) / columns))
        group_heights.append(group_label_h + rows_needed * tile_h + max(0, rows_needed - 1) * gap)

    canvas_w = margin * 2 + columns * tile_w + (columns - 1) * gap
    canvas_h = margin * 2 + header_h + sum(group_heights) + gap * max(0, len(group_heights) - 1)
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    title = "Representative TVA Score Patches"
    if montage_per_class > 0:
        title += f" (top/bottom {montage_per_class} shown)"
    draw.text((margin, margin), title, fill=(22, 22, 22), font=title_font)
    y = margin + header_h

    for label in label_order:
        label_rows = by_label[label]
        band_color = (235, 245, 255) if label.lower() == "villous" else (248, 237, 232)
        draw.rectangle((margin, y, canvas_w - margin, y + group_label_h), fill=band_color)
        draw.text((margin + 10, y + 5), label.title(), fill=(20, 20, 20), font=label_font)
        y += group_label_h

        for idx, row in enumerate(label_rows):
            col = idx % columns
            row_num = idx // columns
            x = margin + col * (tile_w + gap)
            tile_y = y + row_num * (tile_h + gap)
            image = Image.open(row["patch_png"]).convert("RGB")
            image.thumbnail((thumbnail_size, thumbnail_size), Image.Resampling.LANCZOS)
            thumb = Image.new("RGB", (thumbnail_size, thumbnail_size), (245, 245, 245))
            paste_x = (thumbnail_size - image.width) // 2
            paste_y = (thumbnail_size - image.height) // 2
            thumb.paste(image, (paste_x, paste_y))
            canvas.paste(thumb, (x, tile_y))
            draw.rectangle((x, tile_y, x + tile_w - 1, tile_y + thumbnail_size - 1), outline=(195, 195, 195))

            caption_top = tile_y + thumbnail_size + 4
            sample_text = row["sample_id"]
            score_text = f"score {float(row['score']):+.4f}"
            slide_text = Path(row["slide_relative_path"]).stem
            draw.text((x, caption_top), sample_text, fill=(25, 25, 25), font=text_font)
            draw.text((x, caption_top + 16), score_text, fill=(70, 70, 70), font=small_font)
            truncated_slide = slide_text[:28] + ("..." if len(slide_text) > 28 else "")
            draw.text((x, caption_top + 29), truncated_slide, fill=(90, 90, 90), font=small_font)

        rows_needed = int(np.ceil(len(label_rows) / columns))
        y += rows_needed * tile_h + max(0, rows_needed - 1) * gap + gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    args = parse_args()
    if args.montage_per_class < 0:
        raise ValueError("--montage-per-class must be zero or positive")
    if args.montage_columns < 1:
        raise ValueError("--montage-columns must be positive")
    configure_tiatoolbox_path(args.tiatoolbox_root)

    samples_csv = args.samples_csv.resolve()
    slide_root = args.slide_root.resolve()
    output_dir = args.output_dir.resolve()
    samples = load_samples(samples_csv)

    manifest_rows = extract_patches(
        samples=samples,
        slide_root=slide_root,
        output_dir=output_dir,
        read_resolution=args.read_resolution,
        read_units=args.read_units,
        patch_max_size=args.patch_max_size,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print(f"Dry run OK: {len(manifest_rows)} slide paths resolved.")
        return

    manifest_path = output_dir / args.manifest_name
    write_manifest(manifest_rows, manifest_path)

    montage_path = output_dir / args.montage_name
    make_montage(
        manifest_rows,
        montage_path,
        args.thumbnail_size,
        args.montage_per_class,
        args.montage_columns,
    )

    print(f"Wrote {len(manifest_rows)} patch PNGs under {output_dir / 'patches'}")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote montage: {montage_path}")


if __name__ == "__main__":
    main()
