from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any
import numpy as np
import torch

from .config import load_config, resolve_path
from .data import discover_cases
from .features import ensure_feature_cache
from .memory import build_memory, score_case, fit_score_scaler, combine_score
from .metrics import calibrate_threshold, evaluate_case, aggregate_case_metrics


PRIMARY_METRICS = [
    "pixel_auprc_feature_grid",
    "dice_upsampled_input_grid",
    "recall_feature_grid",
    "false_positive_components_feature_grid",
]


def choose_device(cfg: dict[str, Any]) -> torch.device:
    requested = str(cfg["runtime"].get("device", "auto")).lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    if requested.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    return torch.device(requested)


def build_splits(case_ids: list[str], folds: int, seed: int, validation_fraction: float) -> list[dict[str, Any]]:
    if len(case_ids) < folds:
        raise ValueError(f"Need at least {folds} labeled cases for {folds}-fold CV; found {len(case_ids)}.")
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(sorted(case_ids), dtype=object)
    rng.shuffle(shuffled)
    test_parts = np.array_split(shuffled, folds)
    output = []
    for fold_idx, test_part in enumerate(test_parts):
        test_ids = [str(x) for x in test_part.tolist()]
        dev_ids = [str(x) for x in shuffled.tolist() if str(x) not in set(test_ids)]
        fold_rng = np.random.default_rng(seed + 1000 + fold_idx)
        fold_rng.shuffle(dev_ids)
        n_val = max(1, int(math.ceil(len(dev_ids) * validation_fraction)))
        val_ids = dev_ids[:n_val]
        train_ids = dev_ids[n_val:]
        output.append({
            "fold": fold_idx,
            "train_ids": sorted(train_ids),
            "validation_ids": sorted(val_ids),
            "test_ids": sorted(test_ids),
        })
    return output


def _read_cache(cache_dir: Path, case_id: str) -> dict[str, Any]:
    path = cache_dir / f"{case_id}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Feature cache missing for {case_id}: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def _score_ids(
    case_ids: list[str],
    cache_dir: Path,
    memory: dict[str, Any],
    method: str,
    cfg: dict[str, Any],
    device: torch.device,
    scaler: dict[str, float] | None = None,
) -> tuple[list[dict[str, Any]], list[np.ndarray], list[np.ndarray]]:
    raw_list = []
    scores = []
    labels_low = []
    for case_id in case_ids:
        data = _read_cache(cache_dir, case_id)
        raw = score_case(data["features"], memory, cfg, device)
        raw_list.append(raw)
        if scaler is None and "d_normal" in raw:
            # Raw dual components are scaled after the validation-only calibration pass.
            score = raw["d_normal"]
        else:
            score = combine_score(raw, method, scaler or {}, float(cfg["memory"].get("dual_alpha", 0.5)))
        scores.append(score)
        labels_low.append(data["labels"].astype(np.uint8))
    return raw_list, scores, labels_low


def bootstrap_mean_ci(
    rows: list[dict[str, Any]],
    metric_keys: list[str],
    repetitions: int,
    seed: int,
) -> dict[str, list[float] | None]:
    rng = np.random.default_rng(seed)
    result: dict[str, list[float] | None] = {}
    for key in metric_keys:
        values = np.asarray(
            [float(row[key]) for row in rows if row.get(key) is not None and np.isfinite(row[key])],
            dtype=np.float64,
        )
        if len(values) == 0:
            result[key] = None
            continue
        if len(values) == 1 or repetitions <= 0:
            result[key] = [float(values[0]), float(values[0])]
            continue
        indices = rng.integers(0, len(values), size=(int(repetitions), len(values)))
        means = values[indices].mean(axis=1)
        lo, hi = np.percentile(means, [2.5, 97.5])
        result[key] = [float(lo), float(hi)]
    return result


def paired_bootstrap_difference(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    metric_keys: list[str],
    repetitions: int,
    seed: int,
) -> dict[str, dict[str, float] | None]:
    by_a = {str(row["case_id"]): row for row in rows_a}
    by_b = {str(row["case_id"]): row for row in rows_b}
    common = sorted(set(by_a) & set(by_b))
    rng = np.random.default_rng(seed)
    result: dict[str, dict[str, float] | None] = {}
    for key in metric_keys:
        pairs = [
            (float(by_a[c][key]), float(by_b[c][key]))
            for c in common
            if by_a[c].get(key) is not None and by_b[c].get(key) is not None
            and np.isfinite(by_a[c][key]) and np.isfinite(by_b[c][key])
        ]
        if not pairs:
            result[key] = None
            continue
        diffs = np.asarray([a - b for a, b in pairs], dtype=np.float64)
        if len(diffs) == 1 or repetitions <= 0:
            lo = hi = float(diffs.mean())
        else:
            idx = rng.integers(0, len(diffs), size=(int(repetitions), len(diffs)))
            means = diffs[idx].mean(axis=1)
            lo, hi = np.percentile(means, [2.5, 97.5])
        result[key] = {
            "mean_delta_a_minus_b": float(diffs.mean()),
            "ci95_low": float(lo),
            "ci95_high": float(hi),
            "n_paired_cases": int(len(diffs)),
        }
    return result


def _save_results(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def _budget_jobs(cfg: dict[str, Any], fold_idx: int, smoke: bool) -> list[tuple[str, float, int]]:
    if smoke:
        return [("smoke_100pct", 1.0, int(cfg["seed"] + fold_idx))]
    jobs = []
    for budget in cfg["experiment"].get("budgets", [0.1, 0.2, 0.4, 1.0]):
        jobs.append((f"budget_{int(round(float(budget) * 100)):02d}pct", float(budget), int(cfg["seed"] + fold_idx)))
    stress = cfg["experiment"].get("stress_budget", 0.05)
    repetitions = int(cfg["experiment"].get("stress_repetitions", 3))
    for rep in range(repetitions):
        jobs.append((f"stress_{int(round(float(stress) * 100)):02d}pct_rep{rep + 1}", float(stress),
                     int(cfg["seed"] + fold_idx * 100 + rep + 1)))
    return jobs


def run_experiments(config_path: str, smoke: bool = False, force_features: bool = False) -> dict[str, Any]:
    cfg = load_config(config_path)
    root = Path.cwd()
    data_root = resolve_path(root, cfg["data"]["root"])
    cache_dir = resolve_path(root, cfg["features"]["cache_dir"])
    output_dir = resolve_path(root, cfg["experiment"]["output_dir"])
    split_path = resolve_path(root, cfg["experiment"]["split_file"])
    cases = discover_cases(data_root)
    case_ids = [case.case_id for case in cases]
    case_by_id = {case.case_id: case for case in cases}
    device = choose_device(cfg)
    print(f"[BG-PDM] Dataset: {len(cases)} labeled cases; device: {device}")
    print(f"[BG-PDM] Feature cache: {cache_dir}")

    folds = build_splits(
        case_ids,
        folds=int(cfg["experiment"].get("folds", 5)),
        seed=int(cfg["seed"]),
        validation_fraction=float(cfg["experiment"].get("validation_fraction", 0.1)),
    )
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps({
        "seed": int(cfg["seed"]), "folds": folds,
        "note": "Patient-level splits. Each 3D CT volume and all its slices remain in one partition."
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    if smoke:
        selected_fold = folds[0]
        train_ids = selected_fold["train_ids"][:min(4, len(selected_fold["train_ids"]))]
        val_ids = selected_fold["validation_ids"][:min(2, len(selected_fold["validation_ids"]))]
        test_ids = selected_fold["test_ids"][:min(2, len(selected_fold["test_ids"]))]
        needed = set(train_ids + val_ids + test_ids)
        smoke_cases = [case_by_id[c] for c in sorted(needed)]
        smoke_cfg = copy.deepcopy(cfg)
        smoke_cfg["memory"].update({
            "per_case_normal_cap": 200, "per_case_tumor_cap": 100,
            "patient_quota": 32, "max_candidate_pool_normal": 1000,
            "max_candidate_pool_tumor": 500, "normal_bank_size": 128,
            "tumor_bank_size": 64, "coreset_projection_dim": 16,
            "query_chunk_size": 512,
        })
        smoke_cfg["features"]["batch_size"] = min(4, int(cfg["features"].get("batch_size", 16)))
        methods = ["knn_single_random", "bgpdm"]
        ensure_feature_cache(smoke_cases, cache_dir, smoke_cfg, device, force=force_features)
        jobs = [("smoke_100pct", 1.0, int(cfg["seed"]))]
        fold_jobs = [(0, train_ids, val_ids, test_ids, jobs, smoke_cfg, methods)]
    else:
        ensure_feature_cache(cases, cache_dir, cfg, device, force=force_features)
        methods = list(cfg["experiment"].get("methods", []))
        fold_jobs = []
        for fold in folds:
            fold_jobs.append((
                int(fold["fold"]), fold["train_ids"], fold["validation_ids"], fold["test_ids"],
                _budget_jobs(cfg, int(fold["fold"]), False), cfg, methods
            ))

    results_path = output_dir / "experiment_results.json"
    results: dict[str, Any] = {
        "project": "BG-PDM",
        "status": "running",
        "config_path": str(config_path),
        "seed": int(cfg["seed"]),
        "device": str(device),
        "dataset": "MSD Task10 Colon",
        "task_definition": "low-label / weakly supervised tumor localization; masked-out tissue is not assumed healthy",
        "metric_resolution_note": (
            "Pixel AUROC/AUPRC and AUPRO are computed at the feature grid. "
            "Dice is reported both at feature grid and after score upsampling to input resolution; "
            "HD95/NSD use the feature grid with physical spacing correction."
        ),
        "runs": [],
        "paired_comparisons": [],
        "contamination_stress_runs": [],
        "contamination_paired_comparisons": [],
        "warnings": [],
    }
    if not smoke and len(cases) != int(cfg["data"].get("expected_labeled_cases", 126)):
        results["warnings"].append(
            f"Found {len(cases)} cases, expected {cfg['data'].get('expected_labeled_cases', 126)}."
        )
    _save_results(results_path, results)

    for fold_idx, full_train_ids, val_ids, test_ids, jobs, run_cfg, methods in fold_jobs:
        if not val_ids or not test_ids:
            raise RuntimeError(f"Fold {fold_idx} has an empty validation/test partition.")
        print(f"[BG-PDM] Fold {fold_idx}: train={len(full_train_ids)} val={len(val_ids)} test={len(test_ids)}")
        base_rng = np.random.default_rng(int(cfg["seed"]) + fold_idx * 1009)
        base_permutation = [str(x) for x in np.asarray(full_train_ids, dtype=object)[base_rng.permutation(len(full_train_ids))]]

        for budget_name, budget_fraction, budget_seed in jobs:
            # The 5% stress repetitions use independent patient subsets; ordinary budgets are nested.
            if budget_name.startswith("stress_"):
                local_rng = np.random.default_rng(int(budget_seed))
                train_permutation = [str(x) for x in np.asarray(full_train_ids, dtype=object)[local_rng.permutation(len(full_train_ids))]]
            else:
                train_permutation = base_permutation
            n_train = max(1, int(math.ceil(len(train_permutation) * budget_fraction)))
            selected_train_ids = [str(x) for x in train_permutation[:n_train]]
            print(f"[BG-PDM] {budget_name}: selected {len(selected_train_ids)}/{len(full_train_ids)} train cases")
            entries_for_comparison: dict[str, list[dict[str, Any]]] = {}
            for method_index, method in enumerate(methods):
                start_time = time.time()
                print(f"[BG-PDM] Building method={method}")
                memory = build_memory(
                    selected_train_ids, cache_dir, method, run_cfg,
                    seed=int(budget_seed + method_index * 997), device=device
                )
                raw_val, val_scores_raw, val_labels_low = _score_ids(
                    val_ids, cache_dir, memory, method, run_cfg, device
                )
                scaler = fit_score_scaler(raw_val, method, seed=int(budget_seed + 313))
                val_scores = [
                    combine_score(raw, method, scaler, float(run_cfg["memory"].get("dual_alpha", 0.5)))
                    for raw in raw_val
                ]
                val_data = [_read_cache(cache_dir, cid) for cid in val_ids]
                threshold = calibrate_threshold(
                    val_scores, val_labels_low,
                    candidate_count=int(run_cfg["evaluation"].get("threshold_candidates", 64))
                )

                case_rows = []
                for case_id in test_ids:
                    data = _read_cache(cache_dir, case_id)
                    raw = score_case(data["features"], memory, run_cfg, device)
                    score = combine_score(raw, method, scaler, float(run_cfg["memory"].get("dual_alpha", 0.5)))
                    row = evaluate_case(
                        score_low=score,
                        target_low=data["labels"],
                        target_eval=data["labels_eval"],
                        threshold=threshold,
                        spacing_xyz=tuple(float(v) for v in data["spacing_xyz"]),
                        original_shape_xyz=tuple(int(v) for v in data["original_shape_xyz"]),
                        input_size=int(data["input_size"].item()),
                        nsd_tolerance_mm=float(run_cfg["evaluation"].get("nsd_tolerance_mm", 2.0)),
                        aupro_thresholds=int(run_cfg["evaluation"].get("aupro_thresholds", 80)),
                        aupro_max_fpr=float(run_cfg["evaluation"].get("aupro_max_fpr", 0.30)),
                    )
                    row["case_id"] = case_id
                    case_rows.append(row)

                metric_keys = list(aggregate_case_metrics(case_rows).keys())
                metric_keys = [k for k in metric_keys if k != "n_cases"]
                repetitions = int(run_cfg["evaluation"].get("bootstrap_repetitions", 2000))
                summary = aggregate_case_metrics(case_rows)
                ci = bootstrap_mean_ci(case_rows, metric_keys, repetitions, int(budget_seed + method_index))
                elapsed = time.time() - start_time
                entry = {
                    "fold": int(fold_idx),
                    "budget_name": budget_name,
                    "budget_fraction": float(budget_fraction),
                    "method": method,
                    "n_train_cases": int(len(selected_train_ids)),
                    "n_validation_cases": int(len(val_ids)),
                    "n_test_cases": int(len(test_ids)),
                    "train_case_ids": list(selected_train_ids),
                    "validation_case_ids": list(val_ids),
                    "test_case_ids": list(test_ids),
                    "threshold_from_validation": float(threshold),
                    "score_scaler": scaler,
                    "memory_metadata": memory.get("metadata", {}),
                    "summary": summary,
                    "bootstrap_95_ci": ci,
                    "per_case_metrics": case_rows if bool(run_cfg["runtime"].get("save_per_case_metrics", True)) else [],
                    "runtime_seconds": float(elapsed),
                }
                results["runs"].append(entry)
                entries_for_comparison[method] = case_rows
                _save_results(results_path, results)
                print(
                    f"[BG-PDM] done method={method} "
                    f"Dice(input)={summary['dice_upsampled_input_grid']['mean']} "
                    f"AUPRC={summary['pixel_auprc_feature_grid']['mean']} "
                    f"runtime={elapsed:.1f}s"
                )

            if "bgpdm" in entries_for_comparison:
                for baseline in methods:
                    if baseline == "bgpdm":
                        continue
                    results["paired_comparisons"].append({
                        "fold": int(fold_idx),
                        "budget_name": budget_name,
                        "comparison": "bgpdm_minus_baseline",
                        "method_a": "bgpdm",
                        "method_b": baseline,
                        "metrics": paired_bootstrap_difference(
                            entries_for_comparison["bgpdm"],
                            entries_for_comparison[baseline],
                            PRIMARY_METRICS,
                            int(run_cfg["evaluation"].get("bootstrap_repetitions", 2000)),
                            int(budget_seed + 7000 + methods.index(baseline)),
                        ),
                    })
                _save_results(results_path, results)

    # Controlled contamination stress test: replace a known fraction of the
    # non-tumor bank with training-only tumor prototypes, then evaluate on the
    # same held-out test patients. It is a synthetic robustness test, not an
    # estimate of real-world annotation error rates.
    if not smoke and bool(cfg["experiment"].get("run_contamination_stress", True)):
        stress_methods = list(cfg["experiment"].get(
            "contamination_stress_methods",
            ["patchcore_single_coreset", "dual_unprotected", "bgpdm"]
        ))
        contamination_levels = [float(v) for v in cfg["experiment"].get(
            "contamination_levels", [0.0, 0.01, 0.05, 0.10]
        )]
        for fold_idx, full_train_ids, val_ids, test_ids, _jobs, run_cfg, available_methods in fold_jobs:
            stress_methods_fold = [m for m in stress_methods if m in available_methods]
            if not stress_methods_fold:
                continue
            for level_index, contamination_fraction in enumerate(contamination_levels):
                rows_by_method: dict[str, list[dict[str, Any]]] = {}
                for method_index, method in enumerate(stress_methods_fold):
                    stress_seed = int(cfg["seed"] + fold_idx * 10000 + level_index * 1000 + method_index * 101)
                    print(
                        f"[BG-PDM] Contamination stress fold={fold_idx} "
                        f"method={method} fraction={contamination_fraction:.2%}"
                    )
                    start_time = time.time()
                    memory = build_memory(
                        full_train_ids, cache_dir, method, run_cfg,
                        seed=stress_seed, device=device,
                        contamination_fraction=contamination_fraction,
                    )
                    raw_val, _val_scores_unused, val_labels_low = _score_ids(
                        val_ids, cache_dir, memory, method, run_cfg, device
                    )
                    scaler = fit_score_scaler(raw_val, method, seed=stress_seed + 17)
                    val_scores = [
                        combine_score(raw, method, scaler, float(run_cfg["memory"].get("dual_alpha", 0.5)))
                        for raw in raw_val
                    ]
                    threshold = calibrate_threshold(
                        val_scores, val_labels_low,
                        candidate_count=int(run_cfg["evaluation"].get("threshold_candidates", 64))
                    )
                    case_rows = []
                    for case_id in test_ids:
                        data = _read_cache(cache_dir, case_id)
                        raw = score_case(data["features"], memory, run_cfg, device)
                        score = combine_score(raw, method, scaler, float(run_cfg["memory"].get("dual_alpha", 0.5)))
                        row = evaluate_case(
                            score_low=score,
                            target_low=data["labels"],
                            target_eval=data["labels_eval"],
                            threshold=threshold,
                            spacing_xyz=tuple(float(v) for v in data["spacing_xyz"]),
                            original_shape_xyz=tuple(int(v) for v in data["original_shape_xyz"]),
                            input_size=int(data["input_size"].item()),
                            nsd_tolerance_mm=float(run_cfg["evaluation"].get("nsd_tolerance_mm", 2.0)),
                            aupro_thresholds=int(run_cfg["evaluation"].get("aupro_thresholds", 80)),
                            aupro_max_fpr=float(run_cfg["evaluation"].get("aupro_max_fpr", 0.30)),
                        )
                        row["case_id"] = case_id
                        case_rows.append(row)
                    summary = aggregate_case_metrics(case_rows)
                    metric_keys = [k for k in summary.keys() if k != "n_cases"]
                    ci = bootstrap_mean_ci(
                        case_rows, metric_keys,
                        int(run_cfg["evaluation"].get("bootstrap_repetitions", 2000)),
                        stress_seed + 29,
                    )
                    stress_entry = {
                        "fold": int(fold_idx),
                        "contamination_fraction": float(contamination_fraction),
                        "method": method,
                        "n_train_cases": int(len(full_train_ids)),
                        "n_validation_cases": int(len(val_ids)),
                        "n_test_cases": int(len(test_ids)),
                        "threshold_from_validation": float(threshold),
                        "memory_metadata": memory.get("metadata", {}),
                        "summary": summary,
                        "bootstrap_95_ci": ci,
                        "per_case_metrics": case_rows if bool(run_cfg["runtime"].get("save_per_case_metrics", True)) else [],
                        "runtime_seconds": float(time.time() - start_time),
                    }
                    results["contamination_stress_runs"].append(stress_entry)
                    rows_by_method[method] = case_rows
                    _save_results(results_path, results)

                if "bgpdm" in rows_by_method:
                    for baseline in stress_methods_fold:
                        if baseline == "bgpdm":
                            continue
                        results["contamination_paired_comparisons"].append({
                            "fold": int(fold_idx),
                            "contamination_fraction": float(contamination_fraction),
                            "comparison": "bgpdm_minus_baseline",
                            "method_a": "bgpdm",
                            "method_b": baseline,
                            "metrics": paired_bootstrap_difference(
                                rows_by_method["bgpdm"], rows_by_method[baseline], PRIMARY_METRICS,
                                int(cfg["evaluation"].get("bootstrap_repetitions", 2000)),
                                int(cfg["seed"] + fold_idx * 10000 + level_index * 1000 + 9000),
                            ),
                        })
                    _save_results(results_path, results)

    results["status"] = "completed"
    results["completed_at_unix"] = int(time.time())
    _save_results(results_path, results)
    print(f"[BG-PDM] Results written locally to {results_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run patient-level BG-PDM experiments.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--smoke", action="store_true", help="Run a small pipeline check on a few cases.")
    parser.add_argument("--force-features", action="store_true", help="Regenerate cached features.")
    args = parser.parse_args()
    run_experiments(args.config, smoke=args.smoke, force_features=args.force_features)


if __name__ == "__main__":
    main()
