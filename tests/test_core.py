import numpy as np
import torch

from bgpdm.experiment import build_splits
from bgpdm.memory import greedy_coreset, combine_score
from bgpdm.metrics import calibrate_threshold, dice_score, surface_metrics


def test_patient_splits_are_disjoint_and_complete():
    ids = [f"case_{i:03d}" for i in range(25)]
    folds = build_splits(ids, folds=5, seed=7, validation_fraction=0.1)
    assert len(folds) == 5
    for fold in folds:
        train = set(fold["train_ids"])
        val = set(fold["validation_ids"])
        test = set(fold["test_ids"])
        assert train.isdisjoint(val)
        assert train.isdisjoint(test)
        assert val.isdisjoint(test)
        assert train | val | test == set(ids)


def test_greedy_coreset_is_deterministic_and_respects_capacity():
    x = np.random.default_rng(1).normal(size=(40, 12)).astype(np.float16)
    device = torch.device("cpu")
    a = greedy_coreset(x, capacity=10, projection_dim=5, seed=99, device=device)
    b = greedy_coreset(x, capacity=10, projection_dim=5, seed=99, device=device)
    assert a.shape == (10, 12)
    assert np.array_equal(a, b)


def test_dice_and_threshold_calibration():
    target = np.zeros((2, 8, 8), dtype=np.uint8)
    target[:, 2:5, 3:6] = 1
    score = target.astype(np.float32) * 0.8 + 0.1
    threshold = calibrate_threshold([score], [target], candidate_count=16)
    assert 0.1 <= threshold <= 0.9
    assert dice_score(score >= threshold, target) > 0.99


def test_surface_metrics_identical_masks():
    mask = np.zeros((3, 8, 8), dtype=bool)
    mask[:, 2:6, 2:6] = True
    hd95, nsd = surface_metrics(mask, mask, spacing_zyx=(2.0, 1.0, 1.0), tolerance_mm=2.0)
    assert hd95 == 0.0
    assert nsd == 1.0


def test_dual_score_combination_uses_robust_scaler():
    raw = {
        "d_normal": np.array([[[1.0, 2.0]]], dtype=np.float32),
        "log_ratio": np.array([[[0.0, 2.0]]], dtype=np.float32),
    }
    scaler = {
        "d_normal_median": 1.0, "d_normal_iqr": 1.0,
        "log_ratio_median": 0.0, "log_ratio_iqr": 2.0,
    }
    score = combine_score(raw, "bgpdm", scaler, alpha=0.5)
    assert score.shape == (1, 1, 2)
    assert score[0, 0, 1] > score[0, 0, 0]
