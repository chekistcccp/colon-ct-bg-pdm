from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

import nibabel as nib
import numpy as np

from .config import load_config, resolve_path
from .data import discover_cases, load_case
from .metrics import dice_score, surface_metrics


def _require_env(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Optional nnU-Net baseline requires environment variable {name}. "
            "Set nnUNet_raw, nnUNet_preprocessed and nnUNet_results before running."
        )
    path = Path(value).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run(command: list[str], env: dict[str, str]) -> None:
    print("[nnU-Net]", " ".join(command), flush=True)
    subprocess.run(command, env=env, check=True)


def _prepare_dataset(
    dataset_dir: Path,
    cases: list[Any],
    images_dir: Path,
    labels_dir: Path,
) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for name, source in (("imagesTr", images_dir), ("labelsTr", labels_dir)):
        target = dataset_dir / name
        if target.is_symlink():
            if target.resolve() != source.resolve():
                raise RuntimeError(f"{target} already links to a different directory.")
        elif target.exists():
            raise RuntimeError(
                f"{target} already exists and is not a symlink. Refusing to overwrite it; "
                "use a new NNUNET_RAW location or inspect the directory."
            )
        else:
            target.symlink_to(source.resolve(), target_is_directory=True)

    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "tumor": 1},
        "numTraining": len(cases),
        "file_ending": ".nii.gz",
        "training": [
            {
                "image": f"./imagesTr/{case.image_path.name}",
                "label": f"./labelsTr/{case.label_path.name}",
            }
            for case in cases
        ],
    }
    target_json = dataset_dir / "dataset.json"
    if target_json.exists():
        existing = json.loads(target_json.read_text(encoding="utf-8"))
        if existing.get("numTraining") != len(cases) or existing.get("file_ending") != ".nii.gz":
            raise RuntimeError(f"Existing nnU-Net dataset.json does not match this dataset: {target_json}")
    else:
        target_json.write_text(json.dumps(dataset_json, indent=2), encoding="utf-8")


def _evaluate_prediction(pred_path: Path, case_paths: Any, tolerance_mm: float) -> dict[str, Any]:
    case = load_case(case_paths)
    pred_nii = nib.as_closest_canonical(nib.load(str(pred_path)))
    if pred_nii.shape != case.mask.shape or not np.allclose(pred_nii.affine, case.affine, atol=1e-3):
        raise ValueError(
            f"{case.case_id}: nnU-Net prediction geometry does not match the reference mask."
        )
    pred = pred_nii.get_fdata(dtype=np.float32) > 0
    gt = case.mask.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    pred_zyx = pred.transpose(2, 0, 1)
    gt_zyx = gt.transpose(2, 0, 1)
    sx, sy, sz = case.spacing_xyz
    hd95, nsd = surface_metrics(pred_zyx, gt_zyx, (sz, sx, sy), tolerance_mm=tolerance_mm)
    return {
        "case_id": case.case_id,
        "dice": dice_score(pred, gt),
        "precision": float(tp / max(1, tp + fp)),
        "recall": float(tp / max(1, tp + fn)),
        "hd95_mm": hd95,
        "nsd": nsd,
        "predicted_tumor_voxels": int(pred.sum()),
        "reference_tumor_voxels": int(gt.sum()),
    }


def _mean_ci(values: list[float | None]) -> dict[str, float | None]:
    clean = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if len(clean) == 0:
        return {"mean": None, "std": None}
    return {
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=1)) if len(clean) > 1 else 0.0,
    }


def run_baseline(config_path: str) -> dict[str, Any]:
    cfg = load_config(config_path)
    data_root = resolve_path(Path.cwd(), cfg["data"]["root"])
    cases = discover_cases(data_root)
    case_by_id = {case.case_id: case for case in cases}

    nn_raw = _require_env("nnUNet_raw")
    nn_preprocessed = _require_env("nnUNet_preprocessed")
    nn_results = _require_env("nnUNet_results")
    dataset_id = int(os.environ.get("BG_PDM_NNUNET_DATASET_ID", "110"))
    dataset_name = f"Dataset{dataset_id:03d}_ColonBGPDMBaseline"
    dataset_dir = nn_raw / dataset_name
    images_dir = cases[0].image_path.parent
    labels_dir = cases[0].label_path.parent
    _prepare_dataset(dataset_dir, cases, images_dir, labels_dir)

    split_path = resolve_path(Path.cwd(), cfg["experiment"]["split_file"])
    if not split_path.exists():
        raise FileNotFoundError(
            f"Patient-level split file not found: {split_path}. Run the BG-PDM experiment pipeline first."
        )
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    folds = split_payload["folds"]
    env = os.environ.copy()

    _run(["nnUNetv2_plan_and_preprocess", "-d", str(dataset_id), "--verify_dataset_integrity"], env)
    preprocessed_dataset_dir = nn_preprocessed / dataset_name
    if not preprocessed_dataset_dir.exists():
        matches = list(nn_preprocessed.glob(f"{dataset_name}*"))
        if matches:
            preprocessed_dataset_dir = matches[0]
        else:
            raise FileNotFoundError(f"Cannot find nnU-Net preprocessed dataset directory under {nn_preprocessed}")
    splits_final = []
    for fold in folds:
        splits_final.append({"train": fold["train_ids"], "val": fold["validation_ids"]})
    (preprocessed_dataset_dir / "splits_final.json").write_text(
        json.dumps(splits_final, indent=2), encoding="utf-8"
    )

    output_root = resolve_path(Path.cwd(), "outputs/full/nnunet_v2")
    output_root.mkdir(parents=True, exist_ok=True)
    all_fold_results = []
    tolerance = float(cfg["evaluation"].get("nsd_tolerance_mm", 2.0))
    for fold in folds:
        fold_idx = int(fold["fold"])
        start = time.time()
        _run(["nnUNetv2_train", str(dataset_id), "3d_fullres", str(fold_idx)], env)

        fold_root = output_root / f"fold_{fold_idx}"
        input_dir = fold_root / "imagesTs"
        pred_dir = fold_root / "predictions"
        input_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)
        for case_id in fold["test_ids"]:
            image_path = case_by_id[case_id].image_path
            link = input_dir / image_path.name
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(image_path.resolve())
        _run([
            "nnUNetv2_predict", "-i", str(input_dir), "-o", str(pred_dir),
            "-d", str(dataset_id), "-c", "3d_fullres", "-f", str(fold_idx),
            "-chk", "checkpoint_final.pth", "-npp", "1", "-nps", "1",
        ], env)

        case_rows = []
        for case_id in fold["test_ids"]:
            pred_path = pred_dir / f"{case_id}.nii.gz"
            if not pred_path.exists():
                raise FileNotFoundError(f"nnU-Net prediction missing: {pred_path}")
            case_rows.append(_evaluate_prediction(pred_path, case_by_id[case_id], tolerance))
        fold_entry = {
            "fold": fold_idx,
            "n_train_cases": len(fold["train_ids"]),
            "n_validation_cases": len(fold["validation_ids"]),
            "n_test_cases": len(fold["test_ids"]),
            "per_case_metrics": case_rows,
            "summary": {
                "dice": _mean_ci([r["dice"] for r in case_rows]),
                "precision": _mean_ci([r["precision"] for r in case_rows]),
                "recall": _mean_ci([r["recall"] for r in case_rows]),
                "hd95_mm": _mean_ci([r["hd95_mm"] for r in case_rows]),
                "nsd": _mean_ci([r["nsd"] for r in case_rows]),
            },
            "runtime_seconds": float(time.time() - start),
        }
        all_fold_results.append(fold_entry)
        print(f"[nnU-Net] Fold {fold_idx} finished: Dice={fold_entry['summary']['dice']['mean']}")

    payload = {
        "status": "completed",
        "method": "nnU-Net v2 3d_fullres",
        "dataset": "MSD Task10 Colon",
        "dataset_id": dataset_id,
        "split_source": str(split_path),
        "note": "Supervised 100% label reference on the same outer patient-level folds; test labels are used only for evaluation.",
        "folds": all_fold_results,
    }
    results_path = resolve_path(Path.cwd(), cfg["experiment"]["output_dir"]) / "experiment_results.json"
    if results_path.exists():
        combined = json.loads(results_path.read_text(encoding="utf-8"))
    else:
        combined = {"runs": [], "paired_comparisons": []}
    combined["nnunet_v2_reference"] = payload
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"[nnU-Net] Added results to {results_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Optional nnU-Net v2 3D full-resolution reference baseline.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    run_baseline(args.config)


if __name__ == "__main__":
    main()
