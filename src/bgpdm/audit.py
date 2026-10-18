from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from .config import load_config, resolve_path
from .data import discover_cases, load_case


def run_audit(config_path: str) -> dict:
    cfg = load_config(config_path)
    root = Path.cwd()
    data_root = resolve_path(root, cfg["data"]["root"])
    cases = discover_cases(data_root)
    rows = []
    label_values: set[int] = set()
    total_tumor_voxels = 0
    for paths in cases:
        case = load_case(paths)
        unique = [int(v) for v in np.unique(case.mask)]
        label_values.update(unique)
        tumor_voxels = int(case.mask.sum())
        total_tumor_voxels += tumor_voxels
        rows.append({
            "case_id": case.case_id,
            "image_file": paths.image_path.name,
            "label_file": paths.label_path.name,
            "shape_xyz": [int(v) for v in case.image.shape],
            "spacing_xyz_mm": [float(v) for v in case.spacing_xyz],
            "tumor_voxels": tumor_voxels,
            "mask_labels": unique,
            "image_min_hu": float(np.percentile(case.image, 0.1)),
            "image_max_hu": float(np.percentile(case.image, 99.9)),
        })

    expected = int(cfg["data"].get("expected_labeled_cases", 126))
    report = {
        "dataset": "MSD Task10 Colon",
        "data_root": str(data_root),
        "n_labeled_cases": len(cases),
        "expected_labeled_cases": expected,
        "case_count_matches_expected": len(cases) == expected,
        "mask_values_observed": sorted(label_values),
        "total_tumor_voxels": total_tumor_voxels,
        "all_cases_have_tumor_mask": all(row["tumor_voxels"] > 0 for row in rows),
        "cases": rows,
        "warnings": [],
    }
    if len(cases) != expected:
        report["warnings"].append(
            f"Found {len(cases)} labeled image/mask pairs; expected {expected}. "
            "Check whether the dataset is complete or the configured expected count should change."
        )
    if label_values - {0, 1}:
        report["warnings"].append(
            f"Non-binary labels observed: {sorted(label_values)}. The pipeline currently treats label > 0 as tumor."
        )
    if not report["all_cases_have_tumor_mask"]:
        report["warnings"].append("At least one mask is empty; inspect these cases before experiments.")

    out_dir = resolve_path(root, "outputs/audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "data_audit.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_lines = [
        "# MSD Colon 数据审计报告", "",
        f"- 数据根目录：{data_root}",
        f"- 已匹配病例：{len(cases)}",
        f"- 预期病例数：{expected}",
        f"- 病例数匹配：{report['case_count_matches_expected']}",
        f"- 标签值：{report['mask_values_observed']}",
        f"- 肿瘤标注体素总数：{total_tumor_voxels:,}", "",
        "## 警告", ""
    ]
    md_lines.extend([f"- {w}" for w in report["warnings"]] or ["- 无"])
    md_lines += ["", "## 病例摘要", "", "| Case | Shape (X,Y,Z) | Spacing (mm) | Tumor voxels |", "|---|---:|---:|---:|"]
    for row in rows:
        md_lines.append(
            f"| {row['case_id']} | {row['shape_xyz']} | {row['spacing_xyz_mm']} | {row['tumor_voxels']:,} |"
        )
    (out_dir / "data_audit.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[BG-PDM] Audited {len(cases)} cases. Report: {json_path}")
    for warning in report["warnings"]:
        print(f"[BG-PDM][WARNING] {warning}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit MSD Task10 Colon image/mask pairs.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    run_audit(args.config)


if __name__ == "__main__":
    main()
