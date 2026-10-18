from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np
import nibabel as nib


@dataclass(frozen=True)
class CasePaths:
    case_id: str
    image_path: Path
    label_path: Path


@dataclass
class LoadedCase:
    case_id: str
    image: np.ndarray
    mask: np.ndarray
    spacing_xyz: tuple[float, float, float]
    affine: np.ndarray
    image_path: Path
    label_path: Path


def nifti_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def canonical_case_id(path: Path, is_image: bool = False) -> str:
    stem = nifti_stem(path)
    if is_image:
        stem = re.sub(r"_0000$", "", stem)
    return stem


def locate_msd_dirs(data_root: str | Path) -> tuple[Path, Path]:
    root = Path(data_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(
            f"Data root does not exist: {root}\n"
            "Place the extracted MSD Task10 Colon dataset under data/ or configure data.root."
        )

    candidates = [root, root / "Task10_Colon", root / "Task10_Colon" / "Colon"]
    for candidate in candidates:
        images = candidate / "imagesTr"
        labels = candidate / "labelsTr"
        if images.is_dir() and labels.is_dir():
            return images, labels

    # Allow common archive extraction layouts, but only search for directory pairs.
    for images in root.rglob("imagesTr"):
        if images.is_dir() and (images.parent / "labelsTr").is_dir():
            return images, images.parent / "labelsTr"

    raise FileNotFoundError(
        f"Could not find matching imagesTr/labelsTr directories under {root}. "
        "Expected MSD Task10 Colon layout, e.g. data/Task10_Colon/imagesTr and labelsTr."
    )


def discover_cases(data_root: str | Path) -> list[CasePaths]:
    images_dir, labels_dir = locate_msd_dirs(data_root)
    label_files = {
        canonical_case_id(p): p
        for p in labels_dir.iterdir()
        if p.is_file() and (p.name.endswith(".nii.gz") or p.name.endswith(".nii"))
    }
    cases: list[CasePaths] = []
    image_files = [
        p for p in images_dir.iterdir()
        if p.is_file() and (p.name.endswith(".nii.gz") or p.name.endswith(".nii"))
    ]
    for image_path in sorted(image_files):
        case_id = canonical_case_id(image_path, is_image=True)
        label_path = label_files.get(case_id)
        if label_path is None:
            continue
        cases.append(CasePaths(case_id, image_path, label_path))
    if not cases:
        raise RuntimeError(
            f"No image/label pairs found in {images_dir} and {labels_dir}. "
            "Check whether image names use the standard _0000 suffix."
        )
    return cases


def load_case(paths: CasePaths) -> LoadedCase:
    image_nii = nib.as_closest_canonical(nib.load(str(paths.image_path)))
    label_nii = nib.as_closest_canonical(nib.load(str(paths.label_path)))

    if image_nii.shape != label_nii.shape:
        raise ValueError(
            f"{paths.case_id}: image/mask shape mismatch after canonical orientation: "
            f"{image_nii.shape} vs {label_nii.shape}"
        )
    if not np.allclose(image_nii.affine, label_nii.affine, atol=1e-3):
        raise ValueError(
            f"{paths.case_id}: image and mask affine matrices differ after canonical orientation. "
            "Do not silently resample labels; inspect the source data first."
        )

    image = image_nii.get_fdata(dtype=np.float32)
    raw_mask = label_nii.get_fdata(dtype=np.float32)
    if not np.isfinite(image).all():
        raise ValueError(f"{paths.case_id}: CT contains NaN or infinite values")
    if not np.isfinite(raw_mask).all():
        raise ValueError(f"{paths.case_id}: label contains NaN or infinite values")

    unique_labels = np.unique(raw_mask)
    if not np.all(np.isclose(unique_labels, np.round(unique_labels))):
        raise ValueError(f"{paths.case_id}: label contains non-integer values: {unique_labels[:10]}")
    mask = (raw_mask > 0).astype(np.uint8)
    spacing = tuple(float(x) for x in nib.affines.voxel_sizes(image_nii.affine))
    if len(spacing) != 3 or min(spacing) <= 0:
        raise ValueError(f"{paths.case_id}: invalid voxel spacing {spacing}")
    return LoadedCase(
        case_id=paths.case_id,
        image=image,
        mask=mask,
        spacing_xyz=spacing,
        affine=np.asarray(image_nii.affine, dtype=np.float64),
        image_path=paths.image_path,
        label_path=paths.label_path,
    )
