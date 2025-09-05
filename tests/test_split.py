from pathlib import Path

from tvs.cli.split import split_dataset


def test_split_dataset_creates_manifests(tmp_path: Path) -> None:
    source = tmp_path / "src"
    dest = tmp_path / "out"
    source.mkdir()
    for name in [f"slide{i}.db" for i in range(10)]:
        (source / name).write_text("")

    splits = split_dataset(source, dest, val_ratio=0.2, test_ratio=0.2, seed=0)

    # Check file counts
    assert (dest / "train.csv").exists()
    assert (dest / "val.csv").exists()
    assert (dest / "test.csv").exists()

    assert len(splits["train"]) == 6
    assert len(splits["val"]) == 2
    assert len(splits["test"]) == 2

    # Ensure no overlap between splits
    all_ids = [p.stem for v in splits.values() for p in v]
    assert len(all_ids) == len(set(all_ids)) == 10


def test_split_dataset_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    for name in [f"slide{i}.db" for i in range(6)]:
        (source / name).write_text("")

    dest1 = tmp_path / "out1"
    dest2 = tmp_path / "out2"

    splits1 = split_dataset(source, dest1, val_ratio=0.2, test_ratio=0.2, seed=42)
    splits2 = split_dataset(source, dest2, val_ratio=0.2, test_ratio=0.2, seed=42)

    assert [p.stem for p in splits1["train"]] == [p.stem for p in splits2["train"]]
    assert [p.stem for p in splits1["val"]] == [p.stem for p in splits2["val"]]
    assert [p.stem for p in splits1["test"]] == [p.stem for p in splits2["test"]]
