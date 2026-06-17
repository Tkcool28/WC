"""Train a 3-way (H/D/A) classifier on historical WC matches.

This module exposes a simple train/predict API. We use two models:
- Logistic Regression (interpretable baseline, fast, well-calibrated)
- CatBoost multiclass (more flexible, the "real" model)

The model is fit on a feature matrix X (DataFrame) and labels y (Series).
We expose class probabilities (3 columns) so the caller can compute their
own metrics (RPS, Brier, log-loss, calibration).

References:
- Razali et al. 2022: CatBoost + pi-ratings is the top performer in the
  2023 Soccer Prediction Challenge for win/draw/loss probability.
- Berrar et al. 2019: XGBoost with hand-engineered features also performs
  well on similar problems.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler

from soccer_ev_model.pi_ratings import compute_pi_ratings

log = logging.getLogger(__name__)


# Class labels in fixed order
CLASS_LABELS = ("H", "D", "A")  # Home win, Draw, Away win


@dataclass
class TrainedModel:
    """A trained model plus its metadata. Returned by train()."""
    model_type: Literal["logreg", "catboost"]
    feature_columns: list[str]
    scaler: StandardScaler | None  # only used by logreg
    model: object  # sklearn LogisticRegression or CatBoostClassifier

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities. Returns shape (n, 3) in (H, D, A) order."""
        if self.model_type == "logreg":
            Xs = self.scaler.transform(X[self.feature_columns])
        else:
            Xs = X[self.feature_columns]
        return self.model.predict_proba(Xs)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels. Returns array of 'H'/'D'/'A'."""
        probs = self.predict_proba(X)
        idx = probs.argmax(axis=1)
        return np.array([CLASS_LABELS[i] for i in idx])


def train(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: Literal["logreg", "catboost"] = "catboost",
    random_state: int = 42,
) -> TrainedModel:
    """Train a 3-class classifier on the given feature matrix.

    Args:
        X: DataFrame of features. Must include a 'date' column (kept for
           time-based splits, but not used as a feature).
        y: Series of result codes ('H', 'D', 'A').
        model_type: 'logreg' for sklearn LogisticRegression, 'catboost' for
           CatBoost multiclass.
        random_state: seed for reproducibility.
    """
    feature_cols = [c for c in X.columns if c != "date"]
    X_feat = X[feature_cols].copy()

    if model_type == "logreg":
        # Logistic regression: scale features, use multinomial loss.
        # Strong L2 regularization (C=0.1) to fight overfitting on our small
        # training set (~200 matches). With C=1.0 the model memorized the
        # training noise; with C=0.1 it generalizes better.
        # Note: in sklearn >= 1.5, the `multi_class` kwarg was removed because
        # the solver is always multinomial now. We omit it for compatibility.
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_feat)
        model = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            C=0.1,  # stronger regularization
            random_state=random_state,
        )
        model.fit(Xs, y)
        return TrainedModel(
            model_type="logreg",
            feature_columns=feature_cols,
            scaler=scaler,
            model=model,
        )
    elif model_type == "catboost":
        # CatBoost multiclass. Conservative params to avoid overfit on small data.
        # - iterations=100 (not 200) so the model stops earlier
        # - depth=3 (very shallow) so it can't memorize noise
        # - l2_leaf_reg=10 (much higher than default 3) for stronger regularization
        model = CatBoostClassifier(
            iterations=100,
            depth=3,
            learning_rate=0.05,
            l2_leaf_reg=10.0,
            loss_function="MultiClass",
            random_seed=random_state,
            verbose=False,
        )
        model.fit(X_feat, y)
        return TrainedModel(
            model_type="catboost",
            feature_columns=feature_cols,
            scaler=None,
            model=model,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def evaluate(y_true: pd.Series, probs: np.ndarray) -> dict:
    """Compute evaluation metrics for a 3-way probabilistic prediction.

    Args:
        y_true: Series of true labels ('H', 'D', 'A')
        probs: array of shape (n, 3) with columns in (H, D, A) order

    Returns:
        dict with:
            accuracy: fraction of correct predictions
            log_loss: cross-entropy loss
            brier_h: Brier score for home win
            brier_d: Brier score for draw
            brier_a: Brier score for away win
            brier_avg: average of the three Brier scores
            rps: ranked probability score (lower is better)
    """
    y_true = np.array(y_true)
    pred_labels = np.array([CLASS_LABELS[i] for i in probs.argmax(axis=1)])
    accuracy = (pred_labels == y_true).mean()

    # Log loss: convert to one-hot
    label_to_idx = {l: i for i, l in enumerate(CLASS_LABELS)}
    y_idx = np.array([label_to_idx[y] for y in y_true])
    one_hot = np.eye(3)[y_idx]
    # Clip to avoid log(0)
    eps = 1e-15
    log_loss_val = -np.sum(one_hot * np.log(np.clip(probs, eps, 1 - eps))) / len(y_true)

    # Brier per class
    brier_h = brier_score_loss(one_hot[:, 0], probs[:, 0])
    brier_d = brier_score_loss(one_hot[:, 1], probs[:, 1])
    brier_a = brier_score_loss(one_hot[:, 2], probs[:, 2])
    brier_avg = (brier_h + brier_d + brier_a) / 3.0

    # RPS: ranked probability score
    # RPS = sum over k of (cum_pred[k] - cum_actual[k])^2
    # cum_pred[i] = sum of probs up to class i
    cum_pred = np.cumsum(probs, axis=1)
    cum_actual = np.cumsum(one_hot, axis=1)
    rps = np.mean(np.sum((cum_pred - cum_actual) ** 2, axis=1)) / 2.0
    # The /2 normalizes RPS to [0, 1] (without it, it's in [0, 2]).

    return {
        "accuracy": float(accuracy),
        "log_loss": float(log_loss_val),
        "brier_h": float(brier_h),
        "brier_d": float(brier_d),
        "brier_a": float(brier_a),
        "brier_avg": float(brier_avg),
        "rps": float(rps),
        "n": len(y_true),
    }


def report_metrics(metrics: dict, label: str = "") -> str:
    """Format metrics as a one-line summary for printing."""
    return (
        f"{label:>20s}  "
        f"acc={metrics['accuracy']:.3f}  "
        f"log_loss={metrics['log_loss']:.3f}  "
        f"brier_avg={metrics['brier_avg']:.3f}  "
        f"rps={metrics['rps']:.3f}  "
        f"n={metrics['n']}"
    )
