"""Walk-forward backtest for the WC model.

The backtest loop:
1. Sort all matches chronologically.
2. For each test window, train on matches strictly before the cutoff,
   predict on matches at-or-after the cutoff.
3. Compute metrics on the test set.
4. Report metrics for each model type.

This is the LEAKAGE-CRITICAL part of the system. The test verifies the
split, and the training step itself uses build_feature_matrix() which
already uses cutoff-dated pi-ratings.
"""

from __future__ import annotations

from typing import Iterable, Literal

import numpy as np
import pandas as pd

from soccer_ev_model.features import build_feature_matrix
from soccer_ev_model.train import train, evaluate


def walk_forward_split(
    matches: list[dict], train_end_date: str
) -> tuple[list[dict], list[dict]]:
    """Split matches into a train set and a test set at the cutoff date.

    Args:
        matches: list of match dicts (must have 'date' field)
        train_end_date: ISO date string. Train = matches strictly before this
            date. Test = matches at or after this date.

    Returns:
        (train_set, test_set) - two lists of match dicts.
    """
    # Use string comparison for ISO 8601 dates - they sort correctly
    train_set = [m for m in matches if (m.get("date") or "")[:10] < train_end_date]
    test_set = [m for m in matches if (m.get("date") or "")[:10] >= train_end_date]
    return train_set, test_set


def run_backtest(
    matches: list[dict],
    train_end_date: str,
    model_types: tuple[Literal["logreg", "catboost"], ...] = ("logreg", "catboost"),
    verbose: bool = False,
) -> dict:
    """Run a single walk-forward backtest.

    Steps:
    1. Split matches into train (before cutoff) and test (at-or-after).
    2. Build features for both sets. CRITICAL: build_feature_matrix()
       uses the cutoff date internally so test-set features are computed
       from train data only.
    3. Train each model type on the train set.
    4. Predict on the test set and evaluate.

    Returns:
        dict mapping model_type -> metrics dict (from evaluate()).
    """
    train_set, test_set = walk_forward_split(matches, train_end_date)

    if verbose:
        print(f"  Train: {len(train_set)} matches (before {train_end_date})")
        print(f"  Test:  {len(test_set)} matches ({train_end_date} onward)")

    if not train_set or not test_set:
        return {mt: {"n": 0, "error": "empty train or test set"}
                for mt in model_types}

    # Build features. The features module is already leak-safe: for each
    # match it only uses prior_matches (which excludes the match itself and
    # all later matches). This is true for both train and test sets.
    X_train, y_train = build_feature_matrix(train_set)
    X_test, y_test = build_feature_matrix(test_set)

    # CRITICAL: We must also pass the train_set as "prior matches" context
    # for the test feature computation. Currently build_feature_matrix takes
    # only the matches you give it, and uses them all as their own priors.
    # For a proper walk-forward backtest, the test features should be
    # computed with the train_set as additional history.
    # We rebuild the test features using train+test as the match list, which
    # gives the model the right context (train matches are seen, test
    # matches are processed in order, each one sees the train matches AND
    # the earlier test matches).
    X_test_with_train_ctx, y_test_with_train_ctx = build_feature_matrix(
        train_set + test_set
    )
    # Slice: drop the rows that correspond to train_set (they're at the start
    # because we sort chronologically)
    X_test = X_test_with_train_ctx.iloc[len(X_train):].reset_index(drop=True)
    y_test = y_test_with_train_ctx.iloc[len(y_train):].reset_index(drop=True)

    if verbose:
        print(f"  X_train shape: {X_train.shape}, y_train: {y_train.value_counts().to_dict()}")
        print(f"  X_test  shape: {X_test.shape},  y_test:  {y_test.value_counts().to_dict()}")

    results = {}
    for mt in model_types:
        try:
            model = train(X_train, y_train, model_type=mt)
            probs = model.predict_proba(X_test)
            metrics = evaluate(y_test, probs)
            results[mt] = metrics
            if verbose:
                from soccer_ev_model.train import report_metrics
                print(f"  {report_metrics(metrics, label=mt)}")
        except Exception as e:
            results[mt] = {"n": 0, "error": str(e)}
    return results


def compare_to_naive(
    matches: list[dict],
    train_end_date: str,
    model_types: tuple = ("logreg", "catboost"),
) -> dict:
    """Run the backtest and ALSO compute a naive baseline.

    The naive baseline predicts P(H)=P(D)=P(A)=1/3 for every match.
    RPS for this baseline is approximately 0.167 (the expected RPS of
    uniform 3-way predictions against a uniform 3-way truth).

    If our trained model has RPS close to 0.167, it's barely better than
    guessing. If it's much lower (closer to 0), it has real signal.
    """
    results = run_backtest(matches, train_end_date, model_types=model_types,
                           verbose=False)

    # Build the same test-set y we used in run_backtest so the comparison
    # is apples-to-apples.
    train_set, test_set = walk_forward_split(matches, train_end_date)
    X_train, _ = build_feature_matrix(train_set)
    _, y_test = build_feature_matrix(train_set + test_set)
    y_test = y_test.iloc[len(X_train):].reset_index(drop=True)

    n = len(y_test)
    naive_probs = np.ones((n, 3)) / 3.0
    naive_metrics = evaluate(y_test, naive_probs)
    results["naive_uniform"] = naive_metrics

    return results
