from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage as ndi
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm

from .data import CasePaths, load_case


class ResNet18FeatureExtractor(nn.Module):
    """Frozen ImageNet ResNet-18 layer2+layer3 patch features."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        f2 = self.layer2(x)                 # stride 8, 128 channels
        f3 = self.layer3(f2)                # stride 16, 256 channels
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        features = torch.cat([f2, f3], dim=1)
        return F.normalize(features, p=2, dim=1, eps=1e-6)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndi.label(mask)
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == int(counts.argmax())


def _body_mask_from_hu(image_xyz: np.ndarray) -> np.ndarray:
    """A conservative image-only body mask; never uses the tumor label."""
    body = np.zeros(image_xyz.shape, dtype=bool)
    for z in range(image_xyz.shape[2]):
        slice_mask = image_xyz[:, :, z] > -500.0
        slice_mask = ndi.binary_closing(slice_mask, iterations=2)
        slice_mask = ndi.binary_fill_holes(slice_mask)
        body[:, :, z] = _largest_component(slice_mask)
    return body


def _resize_2d(array: np.ndarray, size: int, mode: str) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(array, dtype=np.float32))[None, None]
    kwargs = {"size": (size, size), "mode": mode}
    if mode in ("bilinear", "bicubic"):
        kwargs["align_corners"] = False
    return F.interpolate(tensor, **kwargs)[0, 0].cpu().numpy()


def feature_cache_path(cache_dir: str | Path, case_id: str) -> Path:
    return Path(cache_dir) / f"{case_id}.npz"


def extract_case_features(
    paths: CasePaths,
    cache_dir: str | Path,
    model: ResNet18FeatureExtractor,
    device: torch.device,
    cfg: dict[str, Any],
    force: bool = False,
) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = feature_cache_path(cache_dir, paths.case_id)
    if target.exists() and not force:
        try:
            with np.load(target, allow_pickle=False) as data:
                if int(data["cache_version"].item()) == int(cfg["features"].get("cache_version", 1)):
                    return target
        except Exception:
            # A partial/corrupt cache is safely regenerated.
            pass

    case = load_case(paths)
    image = case.image
    mask = case.mask.astype(bool)
    spacing = case.spacing_xyz
    if mask.sum() == 0:
        raise ValueError(f"{case.case_id}: empty tumor mask; inspect labels before training.")

    # Signed physical distance in millimeters. Array axes are canonical X,Y,Z.
    inside_distance = ndi.distance_transform_edt(mask, sampling=spacing)
    outside_distance = ndi.distance_transform_edt(~mask, sampling=spacing)
    signed_distance = (inside_distance - outside_distance).astype(np.float32)
    body_xyz = _body_mask_from_hu(image)

    lo, hi = cfg["features"].get("intensity_window", [-200.0, 300.0])
    input_size = int(cfg["features"].get("input_size", 224))
    batch_size = max(1, int(cfg["features"].get("batch_size", 16)))
    depth = image.shape[2]
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    feature_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    distance_chunks: list[np.ndarray] = []
    body_chunks: list[np.ndarray] = []
    eval_label_chunks: list[np.ndarray] = []
    feature_hw: tuple[int, int] | None = None

    model.eval()
    for start in range(0, depth, batch_size):
        stop = min(depth, start + batch_size)
        input_batch = []
        for z in range(start, stop):
            z0 = max(0, z - 1)
            z1 = min(depth - 1, z + 1)
            stack = np.stack([image[:, :, z0], image[:, :, z], image[:, :, z1]], axis=0)
            stack = np.clip(stack, float(lo), float(hi))
            stack = (stack - float(lo)) / max(float(hi) - float(lo), 1e-6)
            input_batch.append(stack.astype(np.float32))
        x = torch.from_numpy(np.stack(input_batch, axis=0)).to(device=device, dtype=torch.float32)
        x = F.interpolate(x, size=(input_size, input_size), mode="bilinear", align_corners=False)
        x = (x - mean) / std
        with torch.inference_mode():
            f = model(x)
        if feature_hw is None:
            feature_hw = (int(f.shape[-2]), int(f.shape[-1]))
        feature_chunks.append(f.cpu().to(torch.float16).numpy())

        h, w = feature_hw
        for z in range(start, stop):
            label_chunks.append(_resize_2d(mask[:, :, z].astype(np.float32), h, "nearest").astype(np.uint8))
            distance_chunks.append(_resize_2d(signed_distance[:, :, z], h, "bilinear").astype(np.float16))
            body_chunks.append(_resize_2d(body_xyz[:, :, z].astype(np.float32), h, "nearest").astype(np.uint8))
            eval_label_chunks.append(
                _resize_2d(mask[:, :, z].astype(np.float32), input_size, "nearest").astype(np.uint8)
            )

    features = np.concatenate(feature_chunks, axis=0).astype(np.float16, copy=False)
    labels_low = np.stack(label_chunks, axis=0).astype(np.uint8)
    distance_low = np.stack(distance_chunks, axis=0).astype(np.float16)
    body_low = np.stack(body_chunks, axis=0).astype(np.uint8)
    labels_eval = np.stack(eval_label_chunks, axis=0).astype(np.uint8)

    np.savez_compressed(
        target,
        cache_version=np.asarray(int(cfg["features"].get("cache_version", 1)), dtype=np.int32),
        case_id=np.asarray(case.case_id),
        features=features,
        labels=labels_low,
        signed_distance_mm=distance_low,
        body=body_low,
        labels_eval=labels_eval,
        spacing_xyz=np.asarray(spacing, dtype=np.float32),
        original_shape_xyz=np.asarray(image.shape, dtype=np.int32),
        feature_hw=np.asarray(feature_hw, dtype=np.int32),
        input_size=np.asarray(input_size, dtype=np.int32),
    )
    return target


def ensure_feature_cache(
    cases: list[CasePaths],
    cache_dir: str | Path,
    cfg: dict[str, Any],
    device: torch.device,
    force: bool = False,
) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    model = ResNet18FeatureExtractor(pretrained=bool(cfg["features"].get("pretrained", True))).to(device)
    model.eval()
    for paths in tqdm(cases, desc="Extracting/caching features", unit="case"):
        extract_case_features(paths, cache_dir, model, device, cfg, force=force)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
