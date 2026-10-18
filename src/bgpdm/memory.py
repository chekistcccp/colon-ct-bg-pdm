from __future__ import annotations

from pathlib import Path
from typing import Any
import math
import numpy as np
import torch
import torch.nn.functional as F


DUAL_METHODS = {"dual_unprotected", "boundary_only", "patient_balanced_only", "bgpdm"}
SINGLE_METHODS = {"knn_single_random", "patchcore_single_coreset"}


def _feature_rows(features: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # features: Z,C,H,W; mask: Z,H,W
    rows = features.transpose(0, 2, 3, 1)[mask.astype(bool)]
    if rows.size == 0:
        return np.empty((0, features.shape[1]), dtype=np.float16)
    return np.asarray(rows, dtype=np.float16)


def _sample_rows(rows: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    if len(rows) <= limit:
        return rows
    indices = rng.choice(len(rows), size=limit, replace=False)
    return rows[indices]


def _properties(method: str) -> tuple[bool, bool, bool]:
    """Return (boundary_guard, patient_balanced, needs_tumor_bank)."""
    if method == "bgpdm":
        return True, True, True
    if method == "boundary_only":
        return True, False, True
    if method == "patient_balanced_only":
        return False, True, True
    if method == "dual_unprotected":
        return False, False, True
    if method in SINGLE_METHODS:
        return False, False, False
    raise ValueError(f"Unsupported memory method: {method}")


def _build_candidate_pool(
    case_ids: list[str],
    cache_dir: Path,
    class_name: str,
    boundary_guard: bool,
    patient_balanced: bool,
    cfg: dict[str, Any],
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mem_cfg = cfg["memory"]
    per_case_cap = int(mem_cfg["per_case_normal_cap"] if class_name == "normal"
                       else mem_cfg["per_case_tumor_cap"])
    max_pool = int(mem_cfg["max_candidate_pool_normal"] if class_name == "normal"
                   else mem_cfg["max_candidate_pool_tumor"])
    boundary_mm = float(mem_cfg["boundary_mm"])
    per_case_rows: list[tuple[str, np.ndarray]] = []

    if not case_ids:
        raise ValueError("Cannot build a memory bank from an empty training case list.")

    quota = max(1, min(int(mem_cfg["patient_quota"]), max_pool // max(1, len(case_ids))))
    for case_id in case_ids:
        path = cache_dir / f"{case_id}.npz"
        with np.load(path, allow_pickle=False) as data:
            features = data["features"]
            labels = data["labels"].astype(bool)
            distance = data["signed_distance_mm"].astype(np.float32)
            body = data["body"].astype(bool)

        # Distances are calculated in physical space before downsampling.
        # A feature location is considered core/background using the signed distance.
        if class_name == "normal":
            mask = (distance < (-boundary_mm if boundary_guard else 0.0)) & body
            rows = _feature_rows(features, mask)
        else:
            mask = distance > (boundary_mm if boundary_guard else 0.0)
            rows = _feature_rows(features, mask)

        rows = _sample_rows(rows, per_case_cap, rng)
        if patient_balanced:
            rows = _sample_rows(rows, quota, rng)
        if len(rows):
            per_case_rows.append((case_id, rows))

    if not per_case_rows:
        raise RuntimeError(
            f"No {class_name} candidates for method settings boundary_guard={boundary_guard}. "
            "Check label alignment, body masks, and boundary_mm. The implementation will not "
            "silently insert boundary pixels into a protected tumor bank."
        )

    pool = np.concatenate([rows for _, rows in per_case_rows], axis=0)
    if len(pool) > max_pool:
        if patient_balanced:
            # The per-case quota already equalizes contributions; this last reduction is global.
            indices = rng.choice(len(pool), size=max_pool, replace=False)
        else:
            # Global sampling is intentionally not patient-stratified for ablation baselines.
            indices = rng.choice(len(pool), size=max_pool, replace=False)
        pool = pool[indices]
    return np.asarray(pool, dtype=np.float16)


def greedy_coreset(
    candidates: np.ndarray,
    capacity: int,
    projection_dim: int = 32,
    seed: int = 42,
    device: torch.device | None = None,
) -> np.ndarray:
    """Approximate k-center greedy on a deterministic low-dimensional projection."""
    if candidates.ndim != 2:
        raise ValueError(f"Expected [N,C] candidate matrix, got {candidates.shape}")
    n, channels = candidates.shape
    if n == 0:
        raise ValueError("Cannot select a coreset from zero candidates.")
    if capacity >= n:
        return np.asarray(candidates, dtype=np.float16)
    if capacity <= 0:
        raise ValueError("Coreset capacity must be positive.")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.as_tensor(candidates, dtype=torch.float32, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    dim = min(int(projection_dim), channels)
    projection = torch.randn((channels, dim), generator=generator, device=device) / math.sqrt(dim)
    projected = F.normalize(x @ projection, p=2, dim=1, eps=1e-6)

    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, n))
    selected = [first]
    min_dist = torch.sum((projected - projected[first]) ** 2, dim=1)
    min_dist[first] = -1.0
    for _ in range(1, capacity):
        idx = int(torch.argmax(min_dist).item())
        selected.append(idx)
        distance = torch.sum((projected - projected[idx]) ** 2, dim=1)
        min_dist = torch.minimum(min_dist, distance)
        min_dist[torch.as_tensor(selected, device=device)] = -1.0
    chosen = x[torch.as_tensor(selected, device=device)].to("cpu", dtype=torch.float16).numpy()
    del x, projected, projection, min_dist
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return chosen


def _fit_padim(case_ids: list[str], cache_dir: Path, min_variance: float) -> dict[str, np.ndarray]:
    sum_x = None
    sum_x2 = None
    count = None
    for case_id in case_ids:
        with np.load(cache_dir / f"{case_id}.npz", allow_pickle=False) as data:
            feat = data["features"].astype(np.float32).transpose(0, 2, 3, 1)  # Z,H,W,C
            labels = data["labels"].astype(bool)
            body = data["body"].astype(bool)
            distance = data["signed_distance_mm"].astype(np.float32)
        normal = (~labels) & body & (distance < 0)
        weights = normal.astype(np.float32)
        sx = (feat * weights[..., None]).sum(axis=0)
        sx2 = (feat * feat * weights[..., None]).sum(axis=0)
        cnt = weights.sum(axis=0)
        if sum_x is None:
            sum_x = sx
            sum_x2 = sx2
            count = cnt
        else:
            sum_x += sx
            sum_x2 += sx2
            count += cnt
    assert sum_x is not None and sum_x2 is not None and count is not None
    mean = sum_x / np.maximum(count[..., None], 1.0)
    var = sum_x2 / np.maximum(count[..., None], 1.0) - mean * mean
    var = np.maximum(var, float(min_variance))
    return {"mean": mean.astype(np.float32), "var": var.astype(np.float32)}


def build_memory(
    case_ids: list[str],
    cache_dir: str | Path,
    method: str,
    cfg: dict[str, Any],
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    cache_dir = Path(cache_dir)
    if method == "padim_diagonal":
        padim = _fit_padim(
            case_ids, cache_dir, float(cfg["memory"].get("padim_min_variance", 0.01))
        )
        return {"method": method, "padim": padim, "normal": None, "tumor": None,
                "metadata": {"training_cases": len(case_ids), "normal_candidates": 0, "tumor_candidates": 0}}

    boundary_guard, patient_balanced, needs_tumor = _properties(method)
    normal_pool = _build_candidate_pool(
        case_ids, cache_dir, "normal", boundary_guard, patient_balanced, cfg, seed + 11
    )
    normal_capacity = int(cfg["memory"]["normal_bank_size"])
    if method == "knn_single_random":
        rng = np.random.default_rng(seed + 101)
        normal_bank = _sample_rows(normal_pool, normal_capacity, rng)
    else:
        normal_bank = greedy_coreset(
            normal_pool, normal_capacity,
            projection_dim=int(cfg["memory"]["coreset_projection_dim"]),
            seed=seed + 101, device=device
        )

    tumor_bank = None
    tumor_count = 0
    if needs_tumor:
        tumor_pool = _build_candidate_pool(
            case_ids, cache_dir, "tumor", boundary_guard, patient_balanced, cfg, seed + 23
        )
        tumor_count = int(len(tumor_pool))
        tumor_bank = greedy_coreset(
            tumor_pool, int(cfg["memory"]["tumor_bank_size"]),
            projection_dim=int(cfg["memory"]["coreset_projection_dim"]),
            seed=seed + 211, device=device
        )

    return {
        "method": method,
        "normal": np.asarray(normal_bank, dtype=np.float16),
        "tumor": None if tumor_bank is None else np.asarray(tumor_bank, dtype=np.float16),
        "padim": None,
        "metadata": {
            "training_cases": len(case_ids),
            "normal_candidates": int(len(normal_pool)),
            "tumor_candidates": tumor_count,
            "normal_prototypes": int(len(normal_bank)),
            "tumor_prototypes": 0 if tumor_bank is None else int(len(tumor_bank)),
            "boundary_guard": boundary_guard,
            "patient_balanced": patient_balanced,
        },
    }


def _knn_mean_distance(
    query: np.ndarray,
    bank: np.ndarray,
    k: int,
    chunk_size: int,
    device: torch.device,
) -> np.ndarray:
    if bank is None or len(bank) == 0:
        raise ValueError("Nearest-neighbour scoring requires a non-empty memory bank.")
    q = torch.as_tensor(query, dtype=torch.float32, device=device)
    b = torch.as_tensor(bank, dtype=torch.float32, device=device)
    b = F.normalize(b, p=2, dim=1, eps=1e-6)
    k = max(1, min(int(k), len(bank)))
    outputs = []
    for start in range(0, len(q), max(1, int(chunk_size))):
        qb = F.normalize(q[start:start + chunk_size], p=2, dim=1, eps=1e-6)
        similarities = qb @ b.T
        top_sim = torch.topk(similarities, k=k, dim=1, largest=True, sorted=False).values
        distances = torch.sqrt(torch.clamp(2.0 - 2.0 * top_sim, min=0.0))
        outputs.append(distances.mean(dim=1).detach().cpu().numpy())
    del q, b
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(outputs, axis=0)


def score_case(
    feature_volume: np.ndarray,
    memory: dict[str, Any],
    cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Return raw anomaly-score component maps at feature resolution."""
    z, channels, height, width = feature_volume.shape
    flat = feature_volume.transpose(0, 2, 3, 1).reshape(-1, channels).astype(np.float32)
    method = memory["method"]
    if method == "padim_diagonal":
        mean = memory["padim"]["mean"]
        var = memory["padim"]["var"]
        f = feature_volume.astype(np.float32).transpose(0, 2, 3, 1)
        score = (((f - mean[None]) ** 2) / var[None]).mean(axis=-1)
        return {"score": score.astype(np.float32)}

    d_normal = _knn_mean_distance(
        flat, memory["normal"], int(cfg["memory"]["knn_k"]),
        int(cfg["memory"]["query_chunk_size"]), device
    ).reshape(z, height, width)
    if memory["tumor"] is None:
        return {"score": d_normal.astype(np.float32)}
    d_tumor = _knn_mean_distance(
        flat, memory["tumor"], int(cfg["memory"]["knn_k"]),
        int(cfg["memory"]["query_chunk_size"]), device
    ).reshape(z, height, width)
    eps = float(cfg["memory"].get("eps", 1e-6))
    log_ratio = np.log((d_normal + eps) / (d_tumor + eps))
    return {"d_normal": d_normal.astype(np.float32), "log_ratio": log_ratio.astype(np.float32)}


def fit_score_scaler(
    validation_raw: list[dict[str, np.ndarray]],
    method: str,
    seed: int = 42,
    max_samples: int = 50000,
) -> dict[str, float]:
    """Fit dual-score robust scales using validation images only, without their labels."""
    if method not in DUAL_METHODS:
        return {}
    rng = np.random.default_rng(seed)
    scaler: dict[str, float] = {}
    # Sample each component independently across multiple validation patients.
    # This avoids accidentally calibrating from only the first (often large) volume.
    for key in ("d_normal", "log_ratio"):
        parts: list[np.ndarray] = []
        collected_count = 0
        for item in validation_raw:
            if collected_count >= max_samples:
                break
            arr = item[key].ravel()
            take = min(len(arr), max_samples - collected_count)
            if take <= 0:
                continue
            if len(arr) > take:
                arr = arr[rng.choice(len(arr), size=take, replace=False)]
            parts.append(arr.astype(np.float32))
            collected_count += len(arr)
        values = np.concatenate(parts) if parts else np.asarray([0.0, 1.0])
        median = float(np.median(values))
        q25, q75 = np.percentile(values, [25, 75])
        iqr = float(q75 - q25)
        scaler[f"{key}_median"] = median
        scaler[f"{key}_iqr"] = max(iqr, 1e-6)
    return scaler


def combine_score(raw: dict[str, np.ndarray], method: str, scaler: dict[str, float], alpha: float) -> np.ndarray:
    if method in DUAL_METHODS:
        dn = (raw["d_normal"] - scaler["d_normal_median"]) / scaler["d_normal_iqr"]
        ratio = (raw["log_ratio"] - scaler["log_ratio_median"]) / scaler["log_ratio_iqr"]
        return (float(alpha) * dn + (1.0 - float(alpha)) * ratio).astype(np.float32)
    return raw["score"].astype(np.float32) if "score" in raw else raw["d_normal"].astype(np.float32)
