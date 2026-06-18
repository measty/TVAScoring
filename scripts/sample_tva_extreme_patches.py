"""Sample representative high-villous and high-tubular TVA patches.

The input score CSVs are expected to have columns:
    x,y,width,height,score

The output CSV includes enough relative path information for
extract_tva_patches.py to find the corresponding .ndpi slide beneath a
separate slide root with the same subfolder structure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SCORE_ROOT = Path(r"E:\HIMICO\leuven_new\glasgow_TVA_scores")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly sample slide-capped patches from the extreme TVA score percentiles."
    )
    parser.add_argument(
        "--score-root",
        type=Path,
        default=DEFAULT_SCORE_ROOT,
        help="Root folder containing score CSVs in slide subfolders.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("representative_tva_patches.csv"),
        help="Path for the sampled output CSV.",
    )
    parser.add_argument(
        "--n-per-class",
        type=int,
        default=100,
        help="Number of villous and tubular examples to sample.",
    )
    parser.add_argument(
        "--max-per-slide",
        type=int,
        default=2,
        help="Maximum number of sampled patches to allow from a single slide per class.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=1.0,
        help="Extreme percentile to use at each tail, e.g. 1.0 for top/bottom 1%%.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    return parser.parse_args()


def load_scores(score_root: Path) -> pd.DataFrame:
    files = sorted(score_root.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {score_root}")

    frames: list[pd.DataFrame] = []
    required_columns = ["x", "y", "width", "height", "score"]

    for csv_path in files:
        df = pd.read_csv(csv_path, usecols=required_columns)
        rel = csv_path.relative_to(score_root)
        rel_parent = rel.parent.as_posix()
        df["csv_relative_path"] = rel.as_posix()
        df["slide_subdir"] = rel_parent
        df["slide_stem"] = csv_path.stem
        df["slide_relative_path"] = (rel.parent / f"{csv_path.stem}.ndpi").as_posix()
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def sample_slide_capped(
    eligible: pd.DataFrame,
    label: str,
    n: int,
    rng: np.random.Generator,
    score_cutoff: float,
    percentile_label: str,
    max_per_slide: int,
) -> pd.DataFrame:
    slide_groups = {
        slide_id: group.index.to_numpy()
        for slide_id, group in eligible.groupby("slide_relative_path", sort=True)
    }
    capacity = sum(min(max_per_slide, len(indices)) for indices in slide_groups.values())
    if capacity < n:
        raise ValueError(
            f"Only {capacity} eligible {label} patches are available with "
            f"--max-per-slide={max_per_slide}; need {n}."
        )

    rows = []
    chosen_by_slide: dict[str, set[int]] = {slide_id: set() for slide_id in slide_groups}

    # Fill one patch per slide before taking second patches. This keeps diversity
    # high while still allowing larger samples from the same extreme percentile pool.
    for pass_index in range(max_per_slide):
        candidate_slides = [
            slide_id
            for slide_id, indices in slide_groups.items()
            if len(indices) > pass_index
        ]
        rng.shuffle(candidate_slides)
        for slide_id in candidate_slides:
            if len(rows) == n:
                break
            already_chosen = chosen_by_slide[slide_id]
            if len(already_chosen) >= max_per_slide:
                continue
            remaining_indices = np.array(
                [index for index in slide_groups[slide_id] if index not in already_chosen]
            )
            if len(remaining_indices) == 0:
                continue
            chosen_index = int(rng.choice(remaining_indices, size=1)[0])
            already_chosen.add(chosen_index)
            rows.append(eligible.loc[chosen_index])
        if len(rows) == n:
            break

    sampled = pd.DataFrame(rows).reset_index(drop=True)
    sampled.insert(0, "class", label)
    sampled.insert(1, "class_sample_index", np.arange(1, len(sampled) + 1))
    sampled.insert(2, "percentile_group", percentile_label)
    sampled.insert(3, "score_cutoff", score_cutoff)
    sampled.insert(4, "max_per_slide", max_per_slide)
    return sampled


def main() -> None:
    args = parse_args()
    score_root = args.score_root.resolve()
    output_csv = args.output_csv.resolve()
    if args.n_per_class < 1:
        raise ValueError("--n-per-class must be positive")
    if args.max_per_slide < 1:
        raise ValueError("--max-per-slide must be positive")
    if not 0 < args.percentile < 50:
        raise ValueError("--percentile must be between 0 and 50")

    all_scores = load_scores(score_root)
    lower_q = args.percentile / 100
    upper_q = 1 - lower_q
    tubular_cutoff = float(all_scores["score"].quantile(lower_q))
    villous_cutoff = float(all_scores["score"].quantile(upper_q))

    rng = np.random.default_rng(args.seed)
    tubular = sample_slide_capped(
        all_scores.loc[all_scores["score"] <= tubular_cutoff],
        "tubular",
        args.n_per_class,
        rng,
        tubular_cutoff,
        f"bottom_{args.percentile:g}_percent",
        args.max_per_slide,
    )
    villous = sample_slide_capped(
        all_scores.loc[all_scores["score"] >= villous_cutoff],
        "villous",
        args.n_per_class,
        rng,
        villous_cutoff,
        f"top_{args.percentile:g}_percent",
        args.max_per_slide,
    )

    sampled = pd.concat([villous, tubular], ignore_index=True)
    sample_ids = [
        f"{label}_{int(index):03d}"
        for label, index in zip(
            sampled["class"].to_numpy(),
            sampled["class_sample_index"].to_numpy(),
            strict=True,
        )
    ]
    sampled.insert(0, "sample_id", sample_ids)
    sampled["selection_seed"] = args.seed
    sampled["score_root"] = str(score_root)

    for column in ["x", "y", "width", "height"]:
        sampled[f"{column}_int"] = sampled[column].round().astype(int)

    ordered_columns = [
        "sample_id",
        "class",
        "class_sample_index",
        "percentile_group",
        "score",
        "score_cutoff",
        "max_per_slide",
        "x",
        "y",
        "width",
        "height",
        "x_int",
        "y_int",
        "width_int",
        "height_int",
        "slide_relative_path",
        "csv_relative_path",
        "slide_subdir",
        "slide_stem",
        "selection_seed",
        "score_root",
    ]
    sampled = sampled[ordered_columns]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(output_csv, index=False)

    print(f"Wrote {len(sampled)} sampled patches to {output_csv}")
    print(
        "Thresholds: "
        f"tubular <= {tubular_cutoff:.9f}, villous >= {villous_cutoff:.9f}"
    )
    print(
        "Eligible unique slides: "
        f"tubular={all_scores.loc[all_scores['score'] <= tubular_cutoff, 'slide_relative_path'].nunique()}, "
        f"villous={all_scores.loc[all_scores['score'] >= villous_cutoff, 'slide_relative_path'].nunique()}"
    )


if __name__ == "__main__":
    main()
