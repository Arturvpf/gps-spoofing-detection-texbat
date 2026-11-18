"""Machine-learning classifiers and leak-free evaluation protocols for GPS spoofing detection.

Implements:
- Standardized scikit-learn `Pipeline`s (`StandardScaler` + Classifier) for:
  1. Support Vector Machine (`SVC`, Linear & RBF kernels via `GridSearchCV`)
  2. Random Forest (`RandomForestClassifier` via `GridSearchCV`)
  3. Multi-Layer Perceptron (`MLPClassifier`, simple feedforward neural network)
- Three complementary evaluation protocols:
  1. Stratified 70/30 split + 5-fold Stratified CV (window-level benchmark).
  2. Temporal Block Split with a 10-second guard gap (eliminates temporal autocorrelation leakage).
  3. Leave-One-Attack-Out (LOAO) cross-validation across DS1-DS6 (zero-shot unseen attack generalization).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .data_loader import FEATURE_COLUMNS


@dataclass
class ModelEvaluation:
    """Comprehensive evaluation metrics and artifacts for a trained classifier."""

    model_name: str
    protocol: str
    best_params: dict[str, Any]
    cv_mean_f1: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    confusion_mat: np.ndarray  # [[TN, FP], [FN, TP]] for Clean=0, Spoofed=1
    per_scenario_metrics: pd.DataFrame
    pipeline: Pipeline
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray


def build_candidate_grids(random_state: int = 42) -> dict[str, tuple[Pipeline, dict[str, list[Any]]]]:
    """Return `(pipeline, param_grid)` definitions for SVM, Random Forest, and MLP."""
    svm_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True, class_weight="balanced", random_state=random_state)),
        ]
    )
    svm_grid = {
        "clf__kernel": ["linear", "rbf"],
        "clf__C": [0.5, 1.0, 5.0, 10.0],
        "clf__gamma": ["scale", 0.1],
    }

    rf_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(class_weight="balanced", random_state=random_state, n_jobs=1)),
        ]
    )
    rf_grid = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [8, 14, None],
        "clf__min_samples_leaf": [1, 3],
    }

    mlp_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                MLPClassifier(
                    max_iter=500,
                    early_stopping=True,
                    validation_fraction=0.15,
                    random_state=random_state,
                ),
            ),
        ]
    )
    mlp_grid = {
        "clf__hidden_layer_sizes": [(64, 32), (32, 16)],
        "clf__alpha": [1e-4, 1e-2],
    }

    return {
        "SVM": (svm_pipe, svm_grid),
        "Random Forest": (rf_pipe, rf_grid),
        "MLP (Neural Net)": (mlp_pipe, mlp_grid),
    }


def _encode_labels(labels: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(labels)
    if arr.dtype.kind in ("i", "u", "b"):
        return arr.astype(np.int32)
    return (pd.Series(arr).astype(str).str.lower() == "spoofed").astype(np.int32).to_numpy()


def _per_scenario_breakdown(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    temp = test_df[["scenario", "regime"]].copy()
    temp["y_true"] = y_true
    temp["y_pred"] = y_pred
    temp["correct"] = (y_true == y_pred).astype(np.float64)

    for (sc, reg), grp in temp.groupby(["scenario", "regime"], sort=False):
        n = len(grp)
        acc = float(grp["correct"].mean())
        spoof_rate = float(grp["y_pred"].mean())
        rows.append(
            {
                "scenario": sc,
                "regime": reg,
                "n_windows": int(n),
                "accuracy": round(acc, 4),
                "predicted_spoof_rate": round(spoof_rate, 4),
            }
        )
    return pd.DataFrame(rows)


def _fit_and_score_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    protocol_name: str,
    random_state: int = 42,
) -> dict[str, ModelEvaluation]:
    X_train = train_df[feature_cols].to_numpy(dtype=np.float64)
    y_train = _encode_labels(train_df["label"])
    X_test = test_df[feature_cols].to_numpy(dtype=np.float64)
    y_test = _encode_labels(test_df["label"])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    grids = build_candidate_grids(random_state=random_state)

    evaluations: dict[str, ModelEvaluation] = {}
    for name, (pipe, param_grid) in grids.items():
        search = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring="f1",
            cv=cv,
            n_jobs=1,
            refit=True,
        )
        search.fit(X_train, y_train)
        best_pipe: Pipeline = search.best_estimator_

        y_pred = best_pipe.predict(X_test)
        y_prob = best_pipe.predict_proba(X_test)[:, 1]

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        evaluations[name] = ModelEvaluation(
            model_name=name,
            protocol=protocol_name,
            best_params={k.replace("clf__", ""): v for k, v in search.best_params_.items()},
            cv_mean_f1=float(search.best_score_),
            accuracy=float(accuracy_score(y_test, y_pred)),
            precision=float(precision_score(y_test, y_pred, zero_division=0)),
            recall=float(recall_score(y_test, y_pred, zero_division=0)),
            f1=float(f1_score(y_test, y_pred, zero_division=0)),
            roc_auc=float(roc_auc_score(y_test, y_prob)),
            pr_auc=float(average_precision_score(y_test, y_prob)),
            confusion_mat=cm,
            per_scenario_metrics=_per_scenario_breakdown(test_df, y_test, y_pred),
            pipeline=best_pipe,
            y_true=y_test,
            y_pred=y_pred,
            y_prob=y_prob,
        )
    return evaluations


def evaluate_stratified_split(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    test_size: float = 0.30,
    random_state: int = 42,
) -> dict[str, ModelEvaluation]:
    """Standard 70/30 stratified train-test split across `(scenario, label)` strata."""
    cols = feature_cols or FEATURE_COLUMNS
    strata = df["scenario"].astype(str) + "__" + df["label"].astype(str)
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=strata,
    )
    return _fit_and_score_models(
        train_df=train_df.reset_index(drop=True),
        test_df=test_df.reset_index(drop=True),
        feature_cols=cols,
        protocol_name="Stratified 70/30",
        random_state=random_state,
    )


def split_temporal_blocks(
    df: pd.DataFrame,
    train_fraction: float = 0.70,
    guard_gap_s: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split each `(scenario, regime)` chronologically with a `guard_gap_s` buffer.

    Eliminates temporal-autocorrelation leakage between adjacent 1-second windows.
    """
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for (_, _), grp in df.sort_values("time_s").groupby(["scenario", "regime"], sort=False):
        t_min = float(grp["time_s"].min())
        t_max = float(grp["time_s"].max())
        span = t_max - t_min
        if span <= 2.0 * guard_gap_s:
            cut = int(len(grp) * train_fraction)
            train_parts.append(grp.iloc[:cut])
            test_parts.append(grp.iloc[cut:])
            continue

        t_train_end = t_min + train_fraction * (span - guard_gap_s)
        t_test_start = t_train_end + guard_gap_s
        train_parts.append(grp[grp["time_s"] <= t_train_end])
        test_parts.append(grp[grp["time_s"] >= t_test_start])

    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(test_parts, ignore_index=True),
    )


def evaluate_temporal_block_split(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    train_fraction: float = 0.70,
    guard_gap_s: float = 10.0,
    random_state: int = 42,
) -> dict[str, ModelEvaluation]:
    """Evaluate models on temporally disjoint train/test blocks separated by a guard gap."""
    cols = feature_cols or FEATURE_COLUMNS
    train_df, test_df = split_temporal_blocks(df, train_fraction=train_fraction, guard_gap_s=guard_gap_s)
    return _fit_and_score_models(
        train_df=train_df,
        test_df=test_df,
        feature_cols=cols,
        protocol_name=f"Temporal Block (gap={guard_gap_s:.0f}s)",
        random_state=random_state,
    )


def evaluate_leave_one_attack_out(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Leave-One-Attack-Out (LOAO) evaluation across spoofing scenarios `ds1`..`ds6`.

    In fold `k`, all rows from attack scenario `ds_k` (plus chronological 30% tail blocks
    of `cleanStatic` and `cleanDynamic`) are held out for testing, while the remaining
    5 spoofing scenarios and 70% clean baselines form the training set.
    """
    cols = feature_cols or FEATURE_COLUMNS
    attack_scenarios = sorted([s for s in df["scenario"].unique() if str(s).lower().startswith("ds")])
    clean_df = df[df["scenario"].isin(["cleanStatic", "cleanDynamic"])]
    clean_train, clean_test = split_temporal_blocks(clean_df, train_fraction=0.70, guard_gap_s=10.0)

    records: list[dict[str, Any]] = []
    for held_out in attack_scenarios:
        other_attacks = df[(df["scenario"] != held_out) & (~df["scenario"].isin(["cleanStatic", "cleanDynamic"]))]
        held_df = df[df["scenario"] == held_out]
        held_spoof_only = held_df[held_df["label"] == "Spoofed"]

        train_fold = pd.concat([clean_train, other_attacks], ignore_index=True)
        test_fold = pd.concat([clean_test, held_df], ignore_index=True)

        X_tr = train_fold[cols].to_numpy(dtype=np.float64)
        y_tr = _encode_labels(train_fold["label"])
        X_te = test_fold[cols].to_numpy(dtype=np.float64)
        y_te = _encode_labels(test_fold["label"])

        X_spoof = held_spoof_only[cols].to_numpy(dtype=np.float64)
        X_clean_te = clean_test[cols].to_numpy(dtype=np.float64)

        models = {
            "SVM (RBF)": Pipeline(
                [("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", C=5.0, probability=True, class_weight="balanced", random_state=random_state))]
            ),
            "Random Forest": Pipeline(
                [("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced", random_state=random_state, n_jobs=1))]
            ),
            "MLP (Neural Net)": Pipeline(
                [("scaler", StandardScaler()), ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400, early_stopping=True, random_state=random_state))]
            ),
        }

        for m_name, pipe in models.items():
            pipe.fit(X_tr, y_tr)
            y_pred = pipe.predict(X_te)
            y_prob = pipe.predict_proba(X_te)[:, 1]
            spoof_recall = float(np.mean(pipe.predict(X_spoof) == 1)) if len(X_spoof) > 0 else float("nan")
            clean_far = float(np.mean(pipe.predict(X_clean_te) == 1)) if len(X_clean_te) > 0 else float("nan")

            records.append(
                {
                    "held_out_scenario": held_out,
                    "attack_type": str(held_df["attack_type"].iloc[0]),
                    "platform": str(held_df["platform"].iloc[0]),
                    "model": m_name,
                    "unseen_attack_recall": round(spoof_recall, 4),
                    "clean_false_alarm_rate": round(clean_far, 4),
                    "fold_f1": round(float(f1_score(y_te, y_pred, zero_division=0)), 4),
                    "fold_roc_auc": round(float(roc_auc_score(y_te, y_prob)), 4),
                }
            )

    return pd.DataFrame(records)


def compute_feature_importance(
    evaluation: ModelEvaluation,
    test_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    n_repeats: int = 12,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute permutation importance (F1 drop) on the held-out test set."""
    cols = feature_cols or FEATURE_COLUMNS
    X_test = test_df[cols].to_numpy(dtype=np.float64)
    y_test = _encode_labels(test_df["label"])

    perm = permutation_importance(
        evaluation.pipeline,
        X_test,
        y_test,
        scoring="f1",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1,
    )
    df_imp = pd.DataFrame(
        {
            "feature": cols,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    return df_imp.reset_index(drop=True)
