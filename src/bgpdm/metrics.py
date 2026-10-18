from __future__ import annotations

from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from sklearn.metrics import average_precision_score, roc_auc_score


def upsample_score_map(score_low: np.ndarray, size: int) -> np.ndarray:
    """Bilinearly upsample a Z,H,W score volume to Z,size,size."""
    tensor = torch.from_numpy(np.asarray(score_low, dtype=np.float32))[:, None]
    result = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
    return result[:, 0].cpu().numpy()


def _safe_auc_ap(scores: np.ndarray, labels: np.ndarray) -> tuple[float | None, float | None]:
    y = labels.astype(np.uint8).ravel()
    s = scores.astype(np.float32).ravel()
    if y.min(initial=0) == y.max(initial=0):
        return None, None
    try:
        auc = float(roc_auc_score(y, s))
        ap = float(average_precision_score(y, s))
        return auc, ap
    except ValueError:
        return None, None


def dice_score(prediction: np.ndarray, target: np.ndarray) -> float:
    pred = prediction.astype(bool)
    gt = target.astype(bool)
    denominator = int(pred.sum() + gt.sum())
    if denominator == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, gt).sum() / denominator)


def _surface(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    return mask & ~ndi.binary_erosion(mask, structure=ndi.generate_binary_structure(mask.ndim, 1), border_value=0)


def surface_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    tolerance_mm: float = 2.0,
) -> tuple[float | None, float | None]:
    pred = prediction.astype(bool)
    gt = target.astype(bool)
    if not pred.any() or not gt.any():
        return None, None
    pred_surface = _surface(pred)
    gt_surface = _surface(gt)
    dist_to_gt = ndi.distance_transform_edt(~gt_surface, sampling=spacing_zyx)[pred_surface]
    dist_to_pred = ndi.distance_transform_edt(~pred_surface, sampling=spacing_zyx)[gt_surface]
    all_distances = np.concatenate([dist_to_gt, dist_to_pred])
    hd95 = float(np.percentile(all_distances, 95))
    nsd = float(
        ((dist_to_gt <= tolerance_mm).sum() + (dist_to_pred <= tolerance_mm).sum())
        / max(1, len(dist_to_gt) + len(dist_to_pred))
    )
    return hd95, nsd


def aupro_score(
    score: np.ndarray,
    target: np.ndarray,
    max_fpr: float = 0.30,
    thresholds: int = 80,
) -> float | None:
    """Approximate 3-D AUPRO on the feature grid, normalized over [0, max_fpr]."""
    gt = target.astype(bool)
    negatives = ~gt
    if not gt.any() or not negatives.any():
        return None
    components, n_components = ndi.label(gt, structure=ndi.generate_binary_structure(3, 1))
    if n_components == 0:
        return None
    quantiles = np.linspace(0.0, 1.0, max(8, int(thresholds)))
    cutoffs = np.unique(np.quantile(score.astype(np.float32), quantiles))
    points: list[tuple[float, float]] = []
    for cutoff in cutoffs[::-1]:
        pred = score >= cutoff
        fpr = float(np.logical_and(pred, negatives).sum() / max(1, negatives.sum()))
        if fpr > max_fpr:
            continue
        overlaps = []
        for component_id in range(1, n_components + 1):
            region = components == component_id
            overlaps.append(float(np.logical_and(pred, region).sum() / max(1, region.sum())))
        points.append((fpr, float(np.mean(overlaps))))
    if len(points) < 2:
        return 0.0
    points.sort(key=lambda item: item[0])
    fprs = np.asarray([p[0] for p in points], dtype=np.float64)
    pros = np.asarray([p[1] for p in points], dtype=np.float64)
    # Collapse duplicate FPR values to keep the integration well-defined.
    unique_fpr, inverse = np.unique(fprs, return_inverse=True)
    if len(unique_fpr) < 2:
        return 0.0
    unique_pro = np.zeros_like(unique_fpr)
    counts = np.zeros_like(unique_fpr)
    for idx, group in enumerate(inverse):
        unique_pro[group] += pros[idx]
        counts[group] += 1
    unique_pro /= np.maximum(counts, 1)
    area = float(np.trapz(unique_pro, unique_fpr))
    return float(np.clip(area / max_fpr, 0.0, 1.0))


def calibrate_threshold(
    validation_scores: list[np.ndarray],
    validation_labels_low: list[np.ndarray],
    candidate_count: int = 64,
) -> float:
    """Choose a single threshold by mean validation Dice at feature-map resolution."""
    if not validation_scores:
        return 0.0
    pooled = np.concatenate([s.ravel() for s in validation_scores]).astype(np.float32)
    quantiles = np.linspace(0.02, 0.995, max(8, int(candidate_count)))
    candidates = np.unique(np.quantile(pooled, quantiles))
    best_threshold = float(candidates[0])
    best_dice = -1.0
    for threshold in candidates:
        per_case = [
            dice_score(score >= threshold, label)
            for score, label in zip(validation_scores, validation_labels_low)
        ]
        mean_dice = float(np.mean(per_case)) if per_case else -1.0
        if mean_dice > best_dice:
            best_dice = mean_dice
            best_threshold = float(threshold)
    return best_threshold


def evaluate_case(
    score_low: np.ndarray,
    target_low: np.ndarray,
    target_eval: np.ndarray,
    threshold: float,
    spacing_xyz: tuple[float, float, float],
    original_shape_xyz: tuple[int, int, int],
    input_size: int = 224,
    nsd_tolerance_mm: float = 2.0,
    aupro_thresholds: int = 80,
    aupro_max_fpr: float = 0.30,
) -> dict[str, Any]:
    """Evaluate scores on the feature grid and thresholded masks at input resolution."""
    auc, ap = _safe_auc_ap(score_low, target_low)
    pro = aupro_score(score_low, target_low, max_fpr=aupro_max_fpr, thresholds=aupro_thresholds)

    pred_low = score_low >= threshold
    dice_low = dice_score(pred_low, target_low)
    target_low_bool = target_low.astype(bool)
    tp = int(np.logical_and(pred_low, target_low_bool).sum())
    fp = int(np.logical_and(pred_low, ~target_low_bool).sum())
    fn = int(np.logical_and(~pred_low, target_low_bool).sum())
    precision = float(tp / max(1, tp + fp))
    recall = float(tp / max(1, tp + fn))
    components, n_fp_components = ndi.label(np.logical_and(pred_low, ~target_low_bool))

    score_eval = upsample_score_map(score_low, int(input_size))
    pred_eval = score_eval >= threshold
    gt_eval = target_eval.astype(bool)
    dice_eval = dice_score(pred_eval, gt_eval)

    # The feature-grid physical spacing is derived from the original NIfTI geometry.
    sx, sy, sz = spacing_xyz
    ox, oy, _oz = original_shape_xyz
    z_count, h, w = target_low.shape
    spacing_zyx = (sz, sx * ox / max(1, h), sy * oy / max(1, w))
    hd95, nsd = surface_metrics(pred_low, target_low, spacing_zyx, tolerance_mm=nsd_tolerance_mm)

    return {
        "pixel_auroc_feature_grid": auc,
        "pixel_auprc_feature_grid": ap,
        "aupro_feature_grid": pro,
        "dice_feature_grid": float(dice_low),
        "dice_upsampled_input_grid": float(dice_eval),
        "precision_feature_grid": precision,
        "recall_feature_grid": recall,
        "hd95_mm_feature_grid": hd95,
        "nsd_feature_grid": nsd,
        "false_positive_components_feature_grid": int(n_fp_components),
        "false_positive_voxels_feature_grid": fp,
        "false_negative_voxels_feature_grid": fn,
        "threshold": float(threshold),
        "feature_grid_shape": [int(z_count), int(h), int(w)],
    }


def aggregate_case_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_keys = [
        "pixel_auroc_feature_grid", "pixel_auprc_feature_grid", "aupro_feature_grid",
        "dice_feature_grid", "dice_upsampled_input_grid", "precision_feature_grid",
        "recall_feature_grid", "hd95_mm_feature_grid", "nsd_feature_grid",
        "false_positive_components_feature_grid", "false_positive_voxels_feature_grid",
        "false_negative_voxels_feature_grid",
    ]
    result: dict[str, Any] = {"n_cases": len(rows)}
    for key in metric_keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None and np.isfinite(row[key])]
        result[key] = {
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else (0.0 if values else None),
            "median": float(np.median(values)) if values else None,
            "n_valid": len(values),
        }
    return result
