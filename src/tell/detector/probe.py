"""Lightweight, dependency-minimal runtime for the frozen Tell linear
probe (Part B, Section 13 of the probe-training-pilot spec).

Loads a SafeTensors weights file (`scaler_mean`, `scaler_scale`, `coef`,
`intercept`) and a companion JSON config (selected layer, threshold,
class ordering, corpus/model/protocol hashes, feature dimension), then
scores one 4096-dimensional activation vector with plain NumPy --
StandardScaler.transform followed by a logistic-regression decision
function and sigmoid, exactly reproducing
`sklearn.pipeline.Pipeline([("scaler", StandardScaler()),
("clf", LogisticRegression())]).predict_proba` for the positive class.

No scikit-learn import anywhere in this module -- it is safe to import
in a minimal runtime environment that only has NumPy and `safetensors`
installed. Never imports torch at module scope; a torch.Tensor passed to
`TellProbe.score` is converted via duck-typing only if scikit-learn/torch
happen to also be present in the caller's environment.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open

EXPECTED_FEATURE_DIM = 4096


class ProbeArtifactError(ValueError):
    """Raised for a malformed, mismatched, or shape/NaN/Inf-invalid probe input or artifact."""


@dataclass(frozen=True)
class ProbeScore:
    probability: float
    threshold: float
    alarm: bool
    layer: int


def _to_numpy_vector(vector: Any) -> np.ndarray:
    if hasattr(vector, "detach") and hasattr(vector, "cpu") and hasattr(vector, "numpy"):
        vector = vector.detach().cpu().numpy()  # torch.Tensor, via duck typing only
    arr = np.asarray(vector, dtype=np.float64)
    return arr


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TellProbe:
    """A frozen, single-layer L2-logistic-regression probe. Immutable
    after `load()` -- there is no fit/update method in this runtime."""

    def __init__(self, *, scaler_mean: np.ndarray, scaler_scale: np.ndarray, coef: np.ndarray, intercept: float, layer: int, threshold: float, class_order: tuple[int, int], config: dict) -> None:
        self._scaler_mean = scaler_mean
        self._scaler_scale = scaler_scale
        self._coef = coef
        self._intercept = float(intercept)
        self.layer = layer
        self.threshold = float(threshold)
        self.class_order = class_order
        self.config = config

    @classmethod
    def load(cls, config_path: str | Path, weights_path: str | Path) -> "TellProbe":
        config_path = Path(config_path)
        weights_path = Path(weights_path)
        if not config_path.exists():
            raise ProbeArtifactError(f"probe config not found: {config_path}")
        if not weights_path.exists():
            raise ProbeArtifactError(f"probe weights not found: {weights_path}")

        config = json.loads(config_path.read_text())

        expected_weights_sha256 = config.get("weights_sha256")
        if expected_weights_sha256 is not None:
            actual = _sha256_file(weights_path)
            if actual != expected_weights_sha256:
                raise ProbeArtifactError(f"weights file hash mismatch: config expects {expected_weights_sha256}, found {actual} (model/hash mismatch)")

        with safe_open(str(weights_path), framework="np") as f:
            keys = set(f.keys())
            required = {"scaler_mean", "scaler_scale", "coef", "intercept"}
            missing = required - keys
            if missing:
                raise ProbeArtifactError(f"weights file missing required tensors: {sorted(missing)}")
            scaler_mean = f.get_tensor("scaler_mean").astype(np.float64)
            scaler_scale = f.get_tensor("scaler_scale").astype(np.float64)
            coef = f.get_tensor("coef").astype(np.float64)
            intercept = float(f.get_tensor("intercept").reshape(-1)[0])

        feature_dim = int(config.get("feature_dimension", EXPECTED_FEATURE_DIM))
        for name, arr in (("scaler_mean", scaler_mean), ("scaler_scale", scaler_scale), ("coef", coef)):
            if arr.reshape(-1).shape[0] != feature_dim:
                raise ProbeArtifactError(f"{name} has {arr.reshape(-1).shape[0]} elements, expected feature_dimension={feature_dim}")
            if not np.all(np.isfinite(arr)):
                raise ProbeArtifactError(f"{name} contains NaN or infinite values")
        if not math.isfinite(intercept):
            raise ProbeArtifactError("intercept is NaN or infinite")

        class_order = tuple(config["class_order"])
        if class_order != (0, 1):
            raise ProbeArtifactError(f"unexpected class_order {class_order}, expected (0, 1) = (clean, attack)")

        return cls(
            scaler_mean=scaler_mean.reshape(-1),
            scaler_scale=scaler_scale.reshape(-1),
            coef=coef.reshape(-1),
            intercept=intercept,
            layer=int(config["selected_layer"]),
            threshold=float(config["threshold"]),
            class_order=class_order,
            config=config,
        )

    def score(self, vector: Any) -> ProbeScore:
        arr = _to_numpy_vector(vector)
        if arr.ndim != 1:
            arr = arr.reshape(-1)
        if arr.shape[0] != self._coef.shape[0]:
            raise ProbeArtifactError(f"activation vector has {arr.shape[0]} dimensions, expected {self._coef.shape[0]}")
        if not np.all(np.isfinite(arr)):
            raise ProbeArtifactError("activation vector contains NaN or infinite values")

        scale = np.where(self._scaler_scale == 0, 1.0, self._scaler_scale)  # sklearn StandardScaler convention: zero-variance features left unscaled
        z = (arr - self._scaler_mean) / scale
        logit = float(np.dot(z, self._coef) + self._intercept)
        probability = 1.0 / (1.0 + math.exp(-logit))
        alarm = probability >= self.threshold
        return ProbeScore(probability=probability, threshold=self.threshold, alarm=alarm, layer=self.layer)


__all__ = ["TellProbe", "ProbeScore", "ProbeArtifactError", "EXPECTED_FEATURE_DIM"]
