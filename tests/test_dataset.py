import pytest

from tvs.data.dataset import PatchDataset


def test_patch_dataset_label_mapping(extractors):
    label_scores = {"T": -1.0, "V": 1.0}
    dataset = PatchDataset(extractors, label_scores)

    total_len = sum(len(ex) for lst in extractors.values() for ex in lst)
    assert len(dataset) == total_len
    assert dataset.class_lengths == {
        k: sum(len(ex) for ex in v) for k, v in extractors.items()
    }

    for idx in range(len(dataset)):
        img, target = dataset[idx]
        class_idx, _, _ = dataset._locate(idx)
        label = dataset.labels[class_idx]
        assert target.item() == pytest.approx(label_scores[label])
        assert img.shape[0] == 3
        assert img.shape[1] == img.shape[2] == 512
