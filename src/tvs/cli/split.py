"""CLI for creating train/val/test dataset splits."""

from __future__ import annotations

import argparse
import csv
import logging
import random
from pathlib import Path

log = logging.getLogger(__name__)


def split_dataset(
    source_dir: Path,
    dest_dir: Path,
    *,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 0,
    ext: str = ".db",
) -> dict[str, list[Path]]:
    """Split slide files into train/val/test and write manifest CSVs.

    Parameters
    ----------
    source_dir:
        Directory containing slide-level files (typically annotation ``.db``
        files).  The stem of each file is treated as the slide identifier.
    dest_dir:
        Directory where ``train.csv``, ``val.csv`` and ``test.csv`` will be
        written.
    val_ratio:
        Fraction of slides assigned to the validation split.
    test_ratio:
        Fraction of slides assigned to the test split.
    seed:
        Random seed controlling the shuffling prior to splitting.
    ext:
        File extension to search for within ``source_dir``.

    Returns
    -------
    dict
        Mapping of split name (``train``, ``val``, ``test``) to the list of
        slide paths assigned to that split.  Useful for testing.
    """

    if val_ratio < 0 or test_ratio < 0:
        raise ValueError("val_ratio and test_ratio must be non-negative")
    if val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio + test_ratio must be < 1")

    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    slide_paths = sorted(source_dir.glob(f"*{ext}"))
    rng = random.Random(seed)
    rng.shuffle(slide_paths)

    n = len(slide_paths)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    n_train = n - n_val - n_test

    splits = {
        "train": slide_paths[:n_train],
        "val": slide_paths[n_train : n_train + n_val],
        "test": slide_paths[n_train + n_val :],
    }
    for split_name, paths in splits.items():
        csv_path = dest_dir / f"{split_name}.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["slide_id", "path"])
            for p in paths:
                writer.writerow([p.stem, p.as_posix()])
    return splits


def build_parser() -> argparse.ArgumentParser:
    """Create an argument parser for the dataset splitting CLI."""

    parser = argparse.ArgumentParser(
        description="Create train/val/test splits at the slide level.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Directory containing slide files (e.g. annotation .dbs).",
    )
    parser.add_argument(
        "dest",
        type=Path,
        help="Output directory for split CSV files.",
    )
    parser.add_argument(
        "--val",
        type=float,
        default=0.15,
        help="Fraction of slides for validation.",
    )
    parser.add_argument(
        "--test",
        type=float,
        default=0.15,
        help="Fraction of slides for testing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for shuffling slides before splitting.",
    )
    parser.add_argument(
        "--ext",
        default=".db",
        help="File extension to search for in the source directory.",
    )
    return parser


def main(args: list[str] | None = None) -> None:
    """Entry point for the ``tvs-split`` command."""

    parser = build_parser()
    ns = parser.parse_args(args=args)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    splits = split_dataset(
        ns.source,
        ns.dest,
        val_ratio=ns.val,
        test_ratio=ns.test,
        seed=ns.seed,
        ext=ns.ext,
    )
    log.info(
        "Wrote %d train / %d val / %d test slides",
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
