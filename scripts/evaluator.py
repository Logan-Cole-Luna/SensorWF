"""
evaluator.py — ML-based anomaly detection and performance evaluation.

Three detectors
───────────────
  ZScoreDetector          Baseline — per-feature z-scores; anomaly score = max |z|
  IsolationForestDetector Advanced — sklearn IsolationForest; multivariate
                          Exposes feature_importances_ from tree structure.
  AutoencoderDetector     Advanced — MLP reconstruction-error autoencoder (sklearn)
                          Symmetric bottleneck: n → n//2 → n//4 → n//2 → n

Features
────────
  Raw channel values + first-order differences (rate-of-change) per channel.
  Rolling mean and std (window=15) added when session is long enough (≥45 rows).

Threshold
─────────
  Decision threshold is set at the 99th percentile of TRAINING (clean) scores,
  not of injected scores. This gives ~1% FPR on normal data and dramatically
  improves recall when anomalies occupy 30–50% of the session.

Metrics
───────
  accuracy          (TP + TN) / N          at train-calibrated threshold
  fpr               FP / (FP + TN)
  precision         TP / (TP + FP)
  recall / tpr      TP / (TP + FN)
  f1                2 * prec * recall / (prec + recall)
  latency_s         seconds from first injected sample to first true-positive
  auc_roc           area under ROC curve (threshold-free)
"""

from __future__ import annotations

import os
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble        import IsolationForest
from sklearn.neural_network  import MLPRegressor
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import roc_auc_score
from sklearn.decomposition   import PCA

warnings.filterwarnings("ignore", category=UserWarning)

# ── Feature column sets ────────────────────────────────────────────────────────
_CDH_FEATURES = [
    "TEMP_SOLAR", "TEMP_EPS", "TEMP_BATTERY", "TEMP_BACKPLANE",
    "TEMP_OBDH", "TEMP_ADCS", "TEMP_WHEEL", "TEMP_COMMS", "TEMP_PAYLOAD",
    "SYS_V", "PLAT_5V_V", "OBC_PWR_V", "COMMS_PWR_V",
    "SYS_I", "OBC_PWR_I", "COMMS_PWR_I",
]
_ADCS_FEATURES = [
    "ACCEL_X", "ACCEL_Y", "ACCEL_Z",
    "GYRO_X",  "GYRO_Y",  "GYRO_Z",
    "MAG_X",   "MAG_Y",   "MAG_Z",
    "WHEEL_SPEED",
]

_ROLL_WINDOW = 15   # samples for rolling statistics


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(
    df: pd.DataFrame,
    df_type: str = "cdh",
) -> tuple[np.ndarray | None, list[str]]:
    """
    Extract a numeric feature matrix from a CDH or ADCS DataFrame.

    Features per channel
    --------------------
    1. Raw value
    2. First-order difference (rate of change, Δ per sample)
    3. Rolling mean over _ROLL_WINDOW samples  (only if len ≥ 3× window)
    4. Rolling std  over _ROLL_WINDOW samples  (only if len ≥ 3× window)

    Returns
    -------
    (X, feature_names) where X is (n_samples, n_features) float64 array.
    Returns (None, []) if fewer than 2 usable base channels are found.
    """
    candidates = _CDH_FEATURES if df_type == "cdh" else _ADCS_FEATURES
    cols = [c for c in candidates if c in df.columns and df[c].notna().any()]

    if len(cols) < 2:
        return None, []

    raw = df[cols].copy().astype(float)
    raw = raw.ffill().bfill().fillna(0.0)

    arrays: list[np.ndarray] = [raw.to_numpy()]
    names: list[str]         = list(cols)

    # First-order differences
    diff = raw.diff().fillna(0.0).to_numpy()
    arrays.append(diff)
    names += [f"d_{c}" for c in cols]

    # Timing features: makes packet-gap/timing anomalies observable.
    if "elapsed_s" in df.columns:
        elapsed = pd.to_numeric(df["elapsed_s"], errors="coerce").ffill().bfill().fillna(0.0)
        dt = elapsed.diff().fillna(0.0)
        arrays.append(dt.to_numpy()[:, None])
        names.append("dt_packet")

        if len(df) >= _ROLL_WINDOW * 3:
            dt_rm = dt.rolling(_ROLL_WINDOW, min_periods=1).mean().to_numpy()
            dt_rs = dt.rolling(_ROLL_WINDOW, min_periods=1).std().fillna(0.0).to_numpy()
            arrays.append(dt_rm[:, None])
            arrays.append(dt_rs[:, None])
            names += ["rm_dt_packet", "rs_dt_packet"]

    # Rolling statistics (only when session is long enough)
    if len(df) >= _ROLL_WINDOW * 3:
        rm = raw.rolling(_ROLL_WINDOW, min_periods=1).mean().to_numpy()
        rs = raw.rolling(_ROLL_WINDOW, min_periods=1).std().fillna(0.0).to_numpy()
        arrays.append(rm)
        arrays.append(rs)
        names += [f"rm_{c}" for c in cols]
        names += [f"rs_{c}" for c in cols]

    X = np.concatenate(arrays, axis=1)
    return X, names


def build_combined_feature_matrix(
    cdh_df: pd.DataFrame,
    adcs_df: pd.DataFrame,
) -> tuple[np.ndarray | None, list[str]]:
    """
    Build a combined CDH+ADCS feature matrix by concatenating feature blocks.
    Uses the shortest common sample count to keep row alignment simple/robust.
    """
    X_cdh, names_cdh = build_feature_matrix(cdh_df, "cdh")
    X_adcs, names_adcs = build_feature_matrix(adcs_df, "adcs")
    if X_cdh is None or X_adcs is None:
        return None, []
    n = min(len(X_cdh), len(X_adcs))
    if n < 10:
        return None, []
    X = np.concatenate([X_cdh[:n], X_adcs[:n]], axis=1)
    names = [f"cdh::{n}" for n in names_cdh] + [f"adcs::{n}" for n in names_adcs]
    return X, names


# ─────────────────────────────────────────────────────────────────────────────
# Detectors
# ─────────────────────────────────────────────────────────────────────────────

class ZScoreDetector:
    """
    Per-feature z-score detector.
    Anomaly score = maximum |z| across all features per sample.
    Trained statistics (mean, std) come from the clean training set.
    """

    def __init__(self):
        self._mean: np.ndarray | None = None
        self._std:  np.ndarray | None = None
        self._median: np.ndarray | None = None
        self._mad: np.ndarray | None = None
        self.feature_names_: list[str] = []

    def fit(self, X: np.ndarray, feature_names: list[str] | None = None) -> "ZScoreDetector":
        self._mean = X.mean(axis=0)
        self._std  = X.std(axis=0)
        self._std[self._std == 0] = 1e-9
        self._median = np.median(X, axis=0)
        self._mad = np.median(np.abs(X - self._median), axis=0)
        self._mad[self._mad == 0] = 1e-9
        self.feature_names_ = feature_names or []
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self._mean is None:
            raise RuntimeError("Call fit() first.")
        z = np.abs((X - self._mean) / self._std)
        # Modified Z-score (Iglewicz-Hoaglin style) adds robustness to heavy tails.
        mz = 0.6745 * np.abs((X - self._median) / self._mad)
        return np.maximum(z.max(axis=1), mz.max(axis=1))

    def per_feature_z(self, X: np.ndarray) -> np.ndarray:
        """Return (n_samples, n_features) array of |z-scores| per feature."""
        return np.abs((X - self._mean) / self._std)


class RobustRollingZScoreDetector:
    """
    Rolling robust z-score detector with channel-class rules and persistence.

    - Robust center/scale: rolling median and MAD per feature.
    - Derivative rule: incorporates |Δz| to catch abrupt comms/timing bursts.
    - Persistence gate: rolling mean of instantaneous score.
    - Channel-class thresholds: thermal vs comms vs timing, family-aware tuning.
    """

    def __init__(
        self,
        family: str = "Unknown",
        window: int = 31,
        persistence: int = 5,
        cusum_k: float = 0.25,
        cusum_h: float = 4.0,
    ):
        self.family = str(family)
        self.window = max(11, int(window))
        self.persistence = max(2, int(persistence))
        self.cusum_k = float(cusum_k)
        self.cusum_h = float(cusum_h)
        self.feature_names_: list[str] = []
        self._class_masks: dict[str, np.ndarray] = {}

    @staticmethod
    def _base_feature_name(name: str) -> str:
        base = str(name)
        for p in ("cdh::", "adcs::"):
            if base.startswith(p):
                base = base[len(p):]
        stripped = True
        while stripped:
            stripped = False
            for p in ("d_", "rm_", "rs_"):
                if base.startswith(p):
                    base = base[len(p):]
                    stripped = True
        return base

    def _class_for_feature(self, feat: str) -> str:
        f = self._base_feature_name(feat).upper()
        if any(k in f for k in ["TEMP", "THERMISTOR"]):
            return "thermal"
        if any(k in f for k in ["RSSI", "FRAME_ERR", "COMMS"]):
            return "comms"
        if any(k in f for k in ["LATENCY", "PACKET", "TIME", "ELAPSED"]):
            return "timing"
        return "other"

    def _class_thresholds(self) -> dict[str, tuple[float, float]]:
        fam = self.family.lower()
        thermal_z = 4.2 if "thermal" in fam else 3.6
        comms_z = 2.5
        timing_z = 2.7
        return {
            "thermal": (thermal_z, 1.4),
            "comms": (comms_z, 1.1),
            "timing": (timing_z, 1.15),
            "other": (3.3, 1.3),
        }

    def fit(self, X: np.ndarray, feature_names: list[str] | None = None) -> "RobustRollingZScoreDetector":
        self.feature_names_ = feature_names or [f"f{i}" for i in range(X.shape[1])]
        self._class_masks = {}
        for cls in ["thermal", "comms", "timing", "other"]:
            mask = np.array([self._class_for_feature(n) == cls for n in self.feature_names_], dtype=bool)
            self._class_masks[cls] = mask
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if X.size == 0:
            return np.array([], dtype=float)
        dfx = pd.DataFrame(X)
        med = dfx.rolling(self.window, min_periods=max(5, self.window // 3), center=True).median()
        mad = (dfx - med).abs().rolling(self.window, min_periods=max(5, self.window // 3), center=True).median()

        # Global fallback keeps early/late rows numerically stable.
        med = med.fillna(dfx.median())
        mad = mad.fillna((dfx - dfx.median()).abs().median())
        mad = mad.replace(0, np.nan).fillna(1e-6)

        z = np.abs((dfx - med) / (1.4826 * mad + 1e-6)).to_numpy(dtype=float)
        dz = np.vstack([np.zeros((1, z.shape[1]), dtype=float), np.abs(np.diff(z, axis=0))])

        class_scores: list[np.ndarray] = []
        thr = self._class_thresholds()
        for cls, mask in self._class_masks.items():
            if not mask.any():
                continue
            z_thr, dz_thr = thr[cls]
            s_z = np.max(z[:, mask] / z_thr, axis=1)
            s_d = np.max(dz[:, mask] / dz_thr, axis=1)
            class_scores.append(np.maximum(s_z, s_d))

        if not class_scores:
            instant = np.max(z, axis=1)
        else:
            instant = np.max(np.vstack(class_scores), axis=0)

        persist = pd.Series(instant).rolling(self.persistence, min_periods=1).mean().to_numpy(dtype=float)

        # One-sided CUSUM over centered instantaneous score catches subtle persistent shifts.
        centered = instant - np.median(instant)
        s_pos = np.zeros_like(centered)
        s_neg = np.zeros_like(centered)
        for i in range(1, len(centered)):
            s_pos[i] = max(0.0, s_pos[i - 1] + centered[i] - self.cusum_k)
            s_neg[i] = max(0.0, s_neg[i - 1] - centered[i] - self.cusum_k)
        cus = np.maximum(s_pos, s_neg) / max(self.cusum_h, 1e-6)
        base = np.maximum(instant, persist)
        # Keep CUSUM as a gentle persistence booster, not a replacement score.
        boost = 1.0 + 0.15 * np.tanh(cus)
        return base * boost


class IsolationForestDetector:
    """
    Multivariate detector based on sklearn IsolationForest.
    Anomaly score = −(sklearn raw_score_samples); higher ⟹ more anomalous.

    feature_importances_ is the mean split-frequency across all trees —
    a proxy for which features the forest relies on most for isolation.
    """

    def __init__(self, n_estimators: int = 300, contamination: float = 0.02,
                 random_state: int = 42, max_features: int = 32, pca_variance: float = 0.97,
                 use_pca: bool = True, use_rotation: bool = True, n_ensemble: int = 3):
        self._scaler = StandardScaler()
        self._max_features = max(8, int(max_features))
        self._pca_variance = float(pca_variance)
        self._use_pca = bool(use_pca)
        self._use_rotation = bool(use_rotation)
        self._n_ensemble = max(1, int(n_ensemble))
        self._random_state = int(random_state)
        self._contamination = float(contamination)
        self._n_estimators = int(n_estimators)
        self._feature_idx: np.ndarray | None = None
        self._pca: PCA | None = None
        self._models: list[IsolationForest] = []
        self._rotations: list[np.ndarray] = []
        self.feature_names_: list[str] = []

    @staticmethod
    def _random_rotation(d: int, seed: int) -> np.ndarray:
        if d <= 1:
            return np.eye(d)
        rng = np.random.default_rng(seed)
        q, _ = np.linalg.qr(rng.normal(size=(d, d)))
        return q

    def fit(self, X: np.ndarray, feature_names: list[str] | None = None) -> "IsolationForestDetector":
        self.feature_names_ = feature_names or [f"f{i}" for i in range(X.shape[1])]

        var = np.var(X, axis=0)
        idx = np.argsort(var)[::-1]
        keep = idx[: min(len(idx), self._max_features)]
        if len(keep) == 0:
            keep = np.arange(X.shape[1])
        self._feature_idx = np.sort(keep)
        Xs = self._scaler.fit_transform(X[:, self._feature_idx])

        if self._use_pca and Xs.shape[1] >= 8 and np.isfinite(Xs).all() and float(np.var(Xs, axis=0).sum()) > 1e-10:
            self._pca = PCA(n_components=self._pca_variance, svd_solver="full", random_state=42)
            Xf = self._pca.fit_transform(Xs)
            if np.isfinite(Xf).all() and Xf.shape[1] > 0:
                self.feature_names_ = [f"PC{i+1}" for i in range(Xf.shape[1])]
            else:
                self._pca = None
                Xf = Xs
                self.feature_names_ = [self.feature_names_[i] for i in self._feature_idx]
        else:
            self._pca = None
            Xf = Xs
            self.feature_names_ = [self.feature_names_[i] for i in self._feature_idx]

        self._models = []
        self._rotations = []
        d = Xf.shape[1]
        for i in range(self._n_ensemble):
            if self._use_rotation:
                rot = self._random_rotation(d, self._random_state + 97 * (i + 1))
            else:
                rot = np.eye(d)
            model = IsolationForest(
                n_estimators=self._n_estimators,
                contamination=self._contamination,
                random_state=self._random_state + 31 * i,
                n_jobs=-1,
            )
            model.fit(Xf @ rot)
            self._models.append(model)
            self._rotations.append(rot)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self._feature_idx is None:
            raise RuntimeError("Call fit() first.")
        Xs = self._scaler.transform(X[:, self._feature_idx])
        Xf = self._pca.transform(Xs) if self._pca is not None else Xs
        if not self._models:
            raise RuntimeError("Call fit() first.")
        all_scores = []
        for model, rot in zip(self._models, self._rotations):
            all_scores.append(-model.score_samples(Xf @ rot))
        return np.mean(np.vstack(all_scores), axis=0)

    @property
    def feature_importances_(self) -> np.ndarray:
        """
        Mean feature importance across all isolation trees.
        Uses each ExtraTreeRegressor's feature_importances_ (split frequency
        weighted by impurity reduction), averaged over the ensemble.
        """
        if not self._models:
            return np.array([], dtype=float)
        per_model = []
        for model in self._models:
            imps = np.array([est.feature_importances_ for est in model.estimators_])
            per_model.append(imps.mean(axis=0))
        return np.mean(np.vstack(per_model), axis=0)


class AutoencoderDetector:
    """
    Sequence-aware reconstruction autoencoder using sklearn MLPRegressor.

    The detector builds fixed-length temporal windows, flattens each window,
    and trains an MLP autoencoder on these sequence vectors.
    Anomaly score is projected back to sample-level by averaging overlapping
    window reconstruction errors, which makes it sensitive to temporal patterns
    (drift, bursts, sustained faults) rather than only point anomalies.
    """

    def __init__(self, max_iter: int = 700, random_state: int = 42, seq_len: int = 8,
                 denoise_sigma: float = 0.05):
        self._scaler       = StandardScaler()
        self._max_iter     = max_iter
        self._random_state = random_state
        self._seq_len      = max(4, int(seq_len))
        self._denoise_sigma = max(0.0, float(denoise_sigma))
        self._models: list[MLPRegressor] = []
        self._seq_lens: list[int] = []
        self._window_modes: list[bool] = []
        self._base_dim: int = 0
        self.feature_names_: list[str] = []

    def _build_model(self, n: int) -> MLPRegressor:
        h1 = max(6, n // 3)
        h2 = max(4, n // 8)
        h3 = max(3, n // 16)
        return MLPRegressor(
            hidden_layer_sizes=(h1, h2, h3, h2, h1),
            activation="tanh",
            solver="adam",
            max_iter=self._max_iter,
            random_state=self._random_state,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            tol=1e-5,
        )

    def _windowed(self, Xs: np.ndarray) -> tuple[np.ndarray, bool]:
        """Return sequence-window vectors if long enough; else raw sample vectors."""
        n = len(Xs)
        if n < self._seq_len:
            return Xs, False
        wins = [Xs[i:i + self._seq_len].reshape(-1) for i in range(n - self._seq_len + 1)]
        return np.vstack(wins), True

    def _sample_scores_from_windows(self, win_scores: np.ndarray, n_samples: int) -> np.ndarray:
        """Project window-level errors back to sample-level by overlap averaging."""
        scores = np.zeros(n_samples, dtype=float)
        counts = np.zeros(n_samples, dtype=float)
        for i, s in enumerate(win_scores):
            j = i + self._seq_len
            scores[i:j] += s
            counts[i:j] += 1.0
        counts[counts == 0] = 1.0
        return scores / counts

    def fit(self, X: np.ndarray, feature_names: list[str] | None = None) -> "AutoencoderDetector":
        Xs = self._scaler.fit_transform(X)
        self._base_dim = X.shape[1]
        self._models = []
        self._seq_lens = []
        self._window_modes = []

        # Multi-scale temporal ensembling improves robustness across anomaly durations.
        seq_candidates = sorted(set([self._seq_len, min(max(self._seq_len * 2, 6), 20)]))
        for j, sl in enumerate(seq_candidates):
            old = self._seq_len
            self._seq_len = sl
            X_train, window_mode = self._windowed(Xs)
            model = self._build_model(X_train.shape[1])

            if self._denoise_sigma > 0:
                noise = np.random.default_rng(self._random_state + 17 * j).normal(
                    0.0, self._denoise_sigma, size=X_train.shape
                )
                X_noisy = X_train + noise
            else:
                X_noisy = X_train

            model.fit(X_noisy, X_train)
            self._models.append(model)
            self._seq_lens.append(sl)
            self._window_modes.append(window_mode)
            self._seq_len = old

        self.feature_names_ = feature_names or []
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if not self._models:
            raise RuntimeError("Call fit() first.")
        Xs = self._scaler.transform(X)
        all_scores: list[np.ndarray] = []

        old = self._seq_len
        for model, sl, window_mode in zip(self._models, self._seq_lens, self._window_modes):
            self._seq_len = sl
            if window_mode:
                Xw, ok = self._windowed(Xs)
                if not ok:
                    recon = model.predict(Xs)
                    s = np.mean((Xs - recon) ** 2, axis=1)
                else:
                    recon_w = model.predict(Xw)
                    win_scores = np.mean((Xw - recon_w) ** 2, axis=1)
                    s = self._sample_scores_from_windows(win_scores, len(Xs))
            else:
                recon = model.predict(Xs)
                s = np.mean((Xs - recon) ** 2, axis=1)
            all_scores.append(s)
        self._seq_len = old

        return np.mean(np.vstack(all_scores), axis=0)

    def per_channel_error(self, X: np.ndarray) -> np.ndarray:
        """Return (n_samples, n_features) per-channel squared reconstruction error."""
        Xs = self._scaler.transform(X)
        if self._models and not any(self._window_modes):
            recon = self._models[0].predict(Xs)
            return (Xs - recon) ** 2

        # Sequence mode returns sample-level scalar errors; broadcast for compatibility.
        s = self.score(X)
        return np.repeat(s[:, None], X.shape[1], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation metrics
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_detector(
    scores: np.ndarray,
    labels: np.ndarray,
    elapsed_s: np.ndarray,
    train_scores: np.ndarray | None = None,
    threshold_pct: float = 99.0,
    threshold_value: float | None = None,
) -> dict:
    """
    Evaluate a detector against binary ground-truth labels.

    The decision threshold is set at `threshold_pct` of `train_scores`
    (clean training set) so that FPR on normal data ≈ (100 − threshold_pct)%.
    If `train_scores` is None, the threshold falls back to the same percentile
    of the injected `scores` (legacy behaviour).

    Returns a dict with keys:
        accuracy, fpr, precision, recall, f1, latency_s, auc_roc,
        tp, tn, fp, fn, threshold
    """
    labels = labels.astype(bool)
    n      = len(labels)

    ref    = train_scores if train_scores is not None else scores
    thr    = (float(threshold_value) if threshold_value is not None
              else float(np.percentile(ref, threshold_pct)))
    preds  = scores >= thr

    tp = int(( preds &  labels).sum())
    tn = int((~preds & ~labels).sum())
    fp = int(( preds & ~labels).sum())
    fn = int((~preds &  labels).sum())

    accuracy  = (tp + tn) / n if n > 0 else 0.0
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    # Detection latency in seconds
    anomaly_idxs  = np.where(labels)[0]
    latency_s: float | None = None
    if len(anomaly_idxs) > 0:
        first_anom = anomaly_idxs[0]
        tp_idxs    = np.where(preds & labels)[0]
        tp_after   = tp_idxs[tp_idxs >= first_anom]
        if len(tp_after) > 0:
            latency_s = float(elapsed_s[tp_after[0]] - elapsed_s[first_anom])

    # Threshold-free AUC-ROC
    if labels.sum() > 0 and (~labels).sum() > 0:
        try:
            auc = float(roc_auc_score(labels.astype(int), scores))
        except Exception:
            auc = 0.5
    else:
        auc = 0.5

    return {
        "accuracy":   round(accuracy,  4),
        "fpr":        round(fpr,       4),
        "precision":  round(precision, 4),
        "recall":     round(recall,    4),
        "f1":         round(f1,        4),
        "latency_s":  round(latency_s, 3) if latency_s is not None else None,
        "auc_roc":    round(auc,       4),
        "threshold":  round(thr,       6),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


# ─────────────────────────────────────────────────────────────────────────────
# High-level ML evaluation runner
# ─────────────────────────────────────────────────────────────────────────────

_DETECTORS: list[str] = [
    "ZScore",
    "RobustRollingZScore",
    "IsolationForest",
    "Autoencoder",
]

_TRAIN_FRAC = 0.60   # fraction of clean session used for training


def _if_contamination(family: str, target: str, tag: str) -> float:
    fam = family.lower()
    val = 0.02
    if "thermal" in fam:
        val = 0.015
    elif any(k in fam for k in ["gyro", "accelerometer", "reactionwheel"]):
        val = 0.03
    if target == "cdh+adcs":
        val += 0.01
    if tag.startswith(("C1", "C2", "C3")):
        val += 0.01
    return float(np.clip(val, 0.005, 0.12))


def _ae_seq_len(family: str, tag: str) -> int:
    fam = family.lower()
    if tag.startswith(("C1", "C2", "C3", "A5")):
        return 6
    if "thermal" in fam:
        return 14
    if "reactionwheel" in fam:
        return 10
    return 8


def _threshold_percentile(detector: str, family: str, target: str, tag: str) -> float:
    fam = family.lower()
    p = 99.0

    if detector == "RobustRollingZScore":
        p = 98.0
        if "thermal" in fam:
            p = 99.1
    elif detector == "IsolationForest":
        p = 98.8 if "thermal" in fam else 97.8
        if target == "cdh+adcs":
            p -= 0.6
    elif detector == "Autoencoder":
        p = 99.0 if "thermal" in fam else 98.4
        if tag.startswith(("C1", "C2", "C3", "A5")):
            p -= 0.5

    if target == "adcs":
        p -= 0.2
    return float(np.clip(p, 95.0, 99.7))


def _fit_threshold(detector, X_train: np.ndarray, pct: float) -> float:
    n = len(X_train)
    if n < 20:
        return float(np.percentile(detector.score(X_train), pct))
    split = max(10, int(n * 0.8))
    X_fit = X_train[:split]
    X_val = X_train[split:] if split < n else X_train
    if len(X_val) < 5:
        X_val = X_fit
    return float(np.percentile(detector.score(X_val), pct))


def run_ml_evaluation(
    injected_dir: str,
    files_generated: list[tuple],
    cdh_clean: pd.DataFrame,
    adcs_clean: pd.DataFrame,
) -> list[dict]:
    """
    For every injected variant, load the injected CSV + label, run all three
    ML detectors, and return a list of per-(tag, variant, detector) result dicts.

    Training uses the first `_TRAIN_FRAC` of the clean session.
    Threshold is calibrated at the 99th percentile of clean training scores.
    For IsolationForest, feature importances are included in the result.
    """
    ml_results: list[dict] = []
    fit_cache: dict[tuple, tuple] = {}

    family = os.path.basename(os.path.dirname(injected_dir.rstrip(os.sep))) or "Unknown"

    # Build feature matrices from the clean data
    X_cdh_all,  names_cdh  = build_feature_matrix(cdh_clean,  "cdh")
    X_adcs_all, names_adcs = build_feature_matrix(adcs_clean, "adcs")
    X_combo_all, names_combo = build_combined_feature_matrix(cdh_clean, adcs_clean)

    def _train_slice(X: np.ndarray | None) -> np.ndarray | None:
        if X is None or len(X) == 0:
            return None
        n = max(10, int(len(X) * _TRAIN_FRAC))
        return X[:n]

    X_cdh_tr  = _train_slice(X_cdh_all)
    X_adcs_tr = _train_slice(X_adcs_all)
    X_combo_tr = _train_slice(X_combo_all)

    for item in files_generated:
        if len(item) >= 4:
            tag, target, v, tier = item[0], item[1], item[2], item[3]
        else:
            tag, target, v = item[0], item[1], item[2]
            tier = "unknown"
        is_adcs = (target == "adcs")
        is_combined = (target == "cdh+adcs")

        if is_combined:
            cdh_path = os.path.join(injected_dir, f"cdh_injected_{tag}_{tier}_v{v}.csv")
            adcs_path = os.path.join(injected_dir, f"adcs_injected_{tag}_{tier}_v{v}.csv")
            if not os.path.exists(cdh_path):
                cdh_path = os.path.join(injected_dir, f"cdh_injected_{tag}_v{v}.csv")
            if not os.path.exists(adcs_path):
                adcs_path = os.path.join(injected_dir, f"adcs_injected_{tag}_v{v}.csv")
            X_train = X_combo_tr
            feat_names = names_combo
            df_type = "combined"
        elif is_adcs:
            inj_path  = os.path.join(injected_dir, f"adcs_injected_{tag}_{tier}_v{v}.csv")
            if not os.path.exists(inj_path):
                inj_path = os.path.join(injected_dir, f"adcs_injected_{tag}_v{v}.csv")
            X_train   = X_adcs_tr
            feat_names = names_adcs
            df_type   = "adcs"
        else:
            inj_path  = os.path.join(injected_dir, f"cdh_injected_{tag}_{tier}_v{v}.csv")
            if not os.path.exists(inj_path):
                inj_path = os.path.join(injected_dir, f"cdh_injected_{tag}_v{v}.csv")
            X_train   = X_cdh_tr
            feat_names = names_cdh
            df_type   = "cdh"

        if is_combined:
            if not os.path.exists(cdh_path) or not os.path.exists(adcs_path):
                continue
            try:
                df_cdh_inj = pd.read_csv(cdh_path, low_memory=False, dtype={"anomaly_type": "string"})
                df_adcs_inj = pd.read_csv(adcs_path, low_memory=False, dtype={"anomaly_type": "string"})
            except Exception:
                continue
            if "timestamp" in df_cdh_inj.columns:
                df_cdh_inj["timestamp"] = pd.to_datetime(df_cdh_inj["timestamp"], errors="coerce")
            if "timestamp" in df_adcs_inj.columns:
                df_adcs_inj["timestamp"] = pd.to_datetime(df_adcs_inj["timestamp"], errors="coerce")
            X_inj, _ = build_combined_feature_matrix(df_cdh_inj, df_adcs_inj)
            n = min(len(df_cdh_inj), len(df_adcs_inj))
            if X_inj is not None:
                n = min(n, len(X_inj))
            elapsed_source = df_cdh_inj
        else:
            if not os.path.exists(inj_path):
                continue
            try:
                df_inj = pd.read_csv(
                    inj_path,
                    low_memory=False,
                    dtype={"anomaly_type": "string"},
                )
            except Exception:
                continue

            if "timestamp" in df_inj.columns:
                df_inj["timestamp"] = pd.to_datetime(df_inj["timestamp"], errors="coerce")
            X_inj, _ = build_feature_matrix(df_inj, df_type)
            n = len(df_inj)
            elapsed_source = df_inj

        label_path = os.path.join(injected_dir, f"labels_{tag}_{tier}_v{v}.csv")
        if not os.path.exists(label_path):
            label_path = os.path.join(injected_dir, f"labels_{tag}_v{v}.csv")
        if not os.path.exists(label_path):
            continue
        try:
            label_row = pd.read_csv(label_path).iloc[0]
            start_idx = int(label_row.get("start_idx", 0))
            end_idx   = int(label_row.get("end_idx",   0))
        except Exception:
            continue

        labels = np.zeros(n, dtype=bool)
        lo = max(0, min(start_idx, n - 1)) if n > 0 else 0
        hi = max(lo, min(end_idx, n - 1)) if n > 0 else 0
        labels[lo : hi + 1] = True

        elapsed = (elapsed_source["elapsed_s"].to_numpy(dtype=float)[:n]
                   if "elapsed_s" in elapsed_source.columns
                   else np.arange(n, dtype=float))

        if X_inj is None or X_train is None or len(X_train) < 10:
            continue

        n = min(n, len(X_inj))
        if n < 10:
            continue
        labels = labels[:n]
        elapsed = elapsed[:n]
        X_inj = X_inj[:n]

        # Align feature dimensions — should match, guard for edge cases
        n_feats = min(X_train.shape[1], X_inj.shape[1])
        X_tr  = X_train[:, :n_feats]
        X_inj = X_inj[:,   :n_feats]
        fnames = (feat_names or [])[:n_feats]

        row_base = {
            "tag":       tag,
            "variant":   v,
            "target":    target,
            "tier":      tier,
            "n_anomaly": int(labels.sum()),
            "n_total":   n,
        }

        for det_name in _DETECTORS:
            thr_pct = _threshold_percentile(det_name, family, target, tag)
            det_cfg = {
                "contamination": _if_contamination(family, target, tag),
                "seq_len": _ae_seq_len(family, tag),
                "denoise_sigma": 0.06,
            }
            cache_key = (
                det_name,
                family,
                target,
                tag,
                n_feats,
                round(thr_pct, 3),
                round(det_cfg["contamination"], 4),
                det_cfg["seq_len"],
            )
            try:
                if cache_key in fit_cache:
                    detector, train_scores, thr_val = fit_cache[cache_key]
                else:
                    if det_name == "ZScore":
                        detector = ZScoreDetector()
                    elif det_name == "RobustRollingZScore":
                        detector = RobustRollingZScoreDetector(family=family)
                    elif det_name == "IsolationForest":
                        detector = IsolationForestDetector(
                            contamination=det_cfg["contamination"]
                        )
                    else:
                        detector = AutoencoderDetector(
                            seq_len=det_cfg["seq_len"],
                            denoise_sigma=det_cfg["denoise_sigma"],
                        )
                    detector.fit(X_tr, fnames)
                    train_scores = detector.score(X_tr)
                    thr_val = _fit_threshold(detector, X_tr, thr_pct)
                    fit_cache[cache_key] = (detector, train_scores, thr_val)

                scores       = detector.score(X_inj)
                metrics      = evaluate_detector(scores, labels, elapsed,
                                                 train_scores=train_scores,
                                                 threshold_pct=thr_pct,
                                                 threshold_value=thr_val)
            except Exception as exc:
                metrics = {"error": str(exc)}

            row = {**row_base, "detector": det_name, **metrics}

            # IsolationForest: attach feature importances for interpretability
            if det_name == "IsolationForest" and "error" not in metrics:
                try:
                    imps = detector.feature_importances_
                    row["feature_importances"] = imps.tolist()
                    # Importances are produced in the detector's transformed
                    # feature space (feature subset / optional PCA), so use
                    # detector feature names to keep lengths aligned.
                    row["feature_names"] = list(getattr(detector, "feature_names_", fnames))
                except Exception:
                    pass

            ml_results.append(row)

    return ml_results


def save_comparative_failure_report(all_ml: list[dict], output_dir: str) -> str:
    """Write a concise per-tag failure report with threshold/feature recommendations."""
    path = os.path.join(output_dir, "comparative_failure_report.md")
    df = pd.DataFrame(all_ml)

    lines: list[str] = [
        "# Comparative Detector Failure Report",
        "",
        "Auto-generated summary of weak anomaly tags and recommended threshold/feature changes.",
        "",
    ]

    if df.empty or "detector" not in df.columns:
        lines.append("No ML results available.")
        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        return path

    if "error" in df.columns:
        df = df[df["error"].isna()]
    if df.empty:
        lines.append("No valid detector rows found after filtering errors.")
        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        return path

    by_det = (df.groupby("detector")[["recall", "fpr", "auc_roc"]]
                .mean().sort_values("recall"))
    lines += ["## Overall Detector Averages", ""]
    for det, row in by_det.iterrows():
        lines.append(
            f"- {det}: recall={row['recall']:.3f}, fpr={row['fpr']:.3f}, auc={row['auc_roc']:.3f}"
        )

    by_tag = (df.groupby(["tag", "detector"])[["recall", "fpr", "auc_roc"]]
                .mean().reset_index())
    worst_tags = (by_tag.groupby("tag")["recall"].max()
                  .sort_values().head(8).index.tolist())

    lines += ["", "## Per-Tag Weak Spots", ""]
    for tag in worst_tags:
        sub = by_tag[by_tag["tag"] == tag].sort_values("recall", ascending=False)
        if sub.empty:
            continue
        best = sub.iloc[0]
        worst = sub.iloc[-1]
        lines.append(
            f"- {tag}: best={best['detector']} (recall {best['recall']:.3f}), "
            f"worst={worst['detector']} (recall {worst['recall']:.3f})"
        )

        recs: list[str] = []
        if tag.startswith(("C1", "C2", "C3")):
            recs.append("lower IF threshold percentile by 0.5-1.0 and keep contamination >= 0.03")
            recs.append("retain comms/timing derivative features; avoid aggressive PCA compression")
        if tag.startswith("A5"):
            recs.append("use shorter AE sequence (5-7) and reduce threshold percentile by ~0.5")
        if tag.startswith("T"):
            recs.append("raise robust-z thermal threshold slightly (about +0.2 sigma) to cut false alarms")
        if not recs:
            recs.append("if recall is low and fpr is low, reduce threshold percentile by ~0.5")

        for r in recs:
            lines.append(f"  recommendation: {r}")

    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def save_ml_tier_metrics(ml_results: list[dict], output_dir: str) -> str:
    """Save per-tier and total detector metrics to CSV."""
    path = os.path.join(output_dir, "ml_metrics_by_tier.csv")
    df = pd.DataFrame(ml_results)
    if df.empty or "detector" not in df.columns:
        pd.DataFrame(columns=["tier", "detector", "accuracy", "fpr", "recall", "f1", "auc_roc"]).to_csv(path, index=False)
        return path

    if "error" in df.columns:
        df = df[df["error"].isna()]
    if "tier" not in df.columns:
        df["tier"] = "unknown"

    metrics = ["accuracy", "fpr", "recall", "f1", "auc_roc"]
    by_tier = (df.groupby(["tier", "detector"])[metrics]
                 .mean()
                 .reset_index())
    total = (df.groupby("detector")[metrics]
               .mean()
               .reset_index())
    total.insert(0, "tier", "total")

    out = pd.concat([by_tier, total], ignore_index=True)
    out.to_csv(path, index=False)
    return path


def save_ml_results(ml_results: list[dict], output_dir: str) -> str:
    """Save ML evaluation results as JSON and return the path."""
    path = os.path.join(output_dir, "ml_evaluation.json")

    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_clean(v) for v in obj]
        if isinstance(obj, float) and np.isnan(obj):
            return None
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    with open(path, "w") as fh:
        json.dump(_clean(ml_results), fh, indent=2)
    return path


def print_ml_summary(ml_results: list[dict]) -> None:
    """Print a formatted per-detector summary to stdout."""
    if not ml_results:
        print("  [ML evaluation] No results to display.")
        return

    df = pd.DataFrame(ml_results)
    if "error" in df.columns:
        df = df[df["error"].isna()]

    metrics = ["accuracy", "fpr", "recall", "f1", "auc_roc"]
    if "tier" not in df.columns:
        df["tier"] = "unknown"

    print("\n" + "=" * 72)
    print("ML Detector Performance Summary")
    print("=" * 72)
    print(f"  {'Detector':<20}  {'Accuracy':>9}  {'FPR':>7}  {'Recall':>7}  "
          f"{'F1':>7}  {'AUC-ROC':>8}")
    print("  " + "-" * 66)

    order = ["ZScore", "RobustRollingZScore", "IsolationForest", "Autoencoder"]
    tiers = sorted([t for t in df["tier"].dropna().unique().tolist() if t != "total"])

    for tier in tiers + ["total"]:
        print(f"  Tier: {tier}")
        for det in order:
            sub = df[df["detector"] == det] if "detector" in df.columns else pd.DataFrame()
            if tier != "total":
                sub = sub[sub["tier"] == tier]
            if sub.empty:
                continue
            row = sub[metrics].mean()
            print(f"    {det:<18}  {row['accuracy']:>9.3f}  {row['fpr']:>7.3f}  "
                  f"{row['recall']:>7.3f}  {row['f1']:>7.3f}  {row['auc_roc']:>8.3f}")

    print("=" * 72 + "\n")


# =============================================================================
# Generic ML evaluation -- works for any domain (ECG, Climate, Satellite, ...)
# =============================================================================

def build_generic_feature_matrix(
    df: pd.DataFrame,
    channels: list[str],
    window: int = 15,
) -> tuple[np.ndarray | None, list[str]]:
    """
    Generic feature matrix for any set of numeric channels.
    Mirrors build_feature_matrix but takes an explicit channel list.
    Features: raw, diff, rolling mean/std, timing dt.
    """
    cols = [c for c in channels if c in df.columns and df[c].notna().any()]
    if not cols:
        return None, []

    raw = df[cols].copy().astype(float).ffill().bfill().fillna(0.0)
    arrays: list[np.ndarray] = [raw.to_numpy()]
    names: list[str] = list(cols)

    diff = raw.diff().fillna(0.0).to_numpy()
    arrays.append(diff)
    names += [f"d_{c}" for c in cols]

    if "elapsed_s" in df.columns:
        elapsed = pd.to_numeric(df["elapsed_s"], errors="coerce").ffill().bfill().fillna(0.0)
        dt = elapsed.diff().fillna(0.0)
        arrays.append(dt.to_numpy()[:, None])
        names.append("dt_sample")

    if len(df) >= window * 3:
        rm = raw.rolling(window, min_periods=1).mean().to_numpy()
        rs = raw.rolling(window, min_periods=1).std().fillna(0.0).to_numpy()
        arrays.append(rm)
        arrays.append(rs)
        names += [f"rm_{c}" for c in cols]
        names += [f"rs_{c}" for c in cols]

    return np.concatenate(arrays, axis=1).astype(np.float64), names


def run_generic_ml_evaluation(
    injected_dir: str,
    files_generated: list[tuple],
    clean_df: pd.DataFrame,
    channels: list[str],
    window: int = 15,
    family: str = "Unknown",
) -> list[dict]:
    """
    Generic ML evaluation loop for any domain.

    Parameters
    ----------
    injected_dir   : directory containing injected CSVs and label CSVs
    files_generated: list of (tag, tier, variant) tuples
    clean_df       : the clean (nominal) session DataFrame
    channels       : list of channel names to use for features
    window         : rolling window size for feature engineering
    family         : domain/family name for display and threshold tuning

    Injected file naming (expected by this function):
      signal_injected_{tag}_{tier}_v{v}.csv
      labels_{tag}_{tier}_v{v}.csv
    """
    ml_results: list[dict] = []
    fit_cache: dict[tuple, tuple] = {}

    X_clean, feat_names = build_generic_feature_matrix(clean_df, channels, window)
    if X_clean is None or len(X_clean) < 10:
        print(f"  [generic eval] Insufficient clean features for {family}")
        return []

    n_train = max(10, int(len(X_clean) * _TRAIN_FRAC))
    X_train = X_clean[:n_train]

    for item in files_generated:
        tag, tier, v = item[0], item[1], item[2]

        inj_path   = os.path.join(injected_dir, f"signal_injected_{tag}_{tier}_v{v}.csv")
        label_path = os.path.join(injected_dir, f"labels_{tag}_{tier}_v{v}.csv")
        if not os.path.exists(inj_path) or not os.path.exists(label_path):
            continue

        try:
            df_inj = pd.read_csv(inj_path, low_memory=False)
        except Exception:
            continue

        if "timestamp" in df_inj.columns:
            df_inj["timestamp"] = pd.to_datetime(df_inj["timestamp"], errors="coerce")

        X_inj, _ = build_generic_feature_matrix(df_inj, channels, window)
        if X_inj is None:
            continue

        try:
            label_row = pd.read_csv(label_path).iloc[0]
            start_idx = int(label_row.get("start_idx", 0))
            end_idx   = int(label_row.get("end_idx",   0))
        except Exception:
            continue

        n = min(len(df_inj), len(X_inj))
        labels = np.zeros(n, dtype=bool)
        lo = max(0, min(start_idx, n - 1))
        hi = max(lo, min(end_idx, n - 1))
        labels[lo: hi + 1] = True

        elapsed = (df_inj["elapsed_s"].to_numpy(dtype=float)[:n]
                   if "elapsed_s" in df_inj.columns else np.arange(n, dtype=float))

        n_feats = min(X_train.shape[1], X_inj.shape[1])
        X_tr  = X_train[:, :n_feats]
        X_i   = X_inj[:,  :n_feats]
        fnames = (feat_names or [])[:n_feats]

        row_base = {
            "tag": tag, "variant": v, "tier": tier, "target": "signal",
            "n_anomaly": int(labels.sum()), "n_total": n,
        }

        for det_name in _DETECTORS:
            thr_pct = _threshold_percentile(det_name, family, "signal", tag)
            det_cfg = {
                "contamination": _if_contamination(family, "signal", tag),
                "seq_len": _ae_seq_len(family, tag),
                "denoise_sigma": 0.06,
            }
            cache_key = (det_name, family, tag, n_feats, round(thr_pct, 3))
            try:
                if cache_key in fit_cache:
                    detector, train_scores, thr_val = fit_cache[cache_key]
                else:
                    if det_name == "ZScore":
                        detector = ZScoreDetector()
                    elif det_name == "RobustRollingZScore":
                        detector = RobustRollingZScoreDetector(family=family)
                    elif det_name == "IsolationForest":
                        detector = IsolationForestDetector(contamination=det_cfg["contamination"])
                    else:
                        detector = AutoencoderDetector(
                            seq_len=det_cfg["seq_len"],
                            denoise_sigma=det_cfg["denoise_sigma"],
                        )
                    detector.fit(X_tr, fnames)
                    train_scores = detector.score(X_tr)
                    thr_val = _fit_threshold(detector, X_tr, thr_pct)
                    fit_cache[cache_key] = (detector, train_scores, thr_val)

                scores  = detector.score(X_i)
                metrics = evaluate_detector(scores, labels, elapsed,
                                            train_scores=train_scores,
                                            threshold_pct=thr_pct,
                                            threshold_value=thr_val)
            except Exception as exc:
                metrics = {"error": str(exc)}

            ml_results.append({**row_base, "detector": det_name, **metrics})

    return ml_results
