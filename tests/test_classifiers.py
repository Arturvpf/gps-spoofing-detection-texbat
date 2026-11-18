"""Tests for leak-free temporal splitting and ML classification pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd

from texbat_spoofing.classifiers import (
    evaluate_stratified_split,
    evaluate_temporal_block_split,
    split_temporal_blocks,
)
from texbat_spoofing.data_loader import FEATURE_COLUMNS


def _make_synthetic_feature_table(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for scenario, is_attack in [("cleanStatic", False), ("cleanDynamic", False), ("ds1", True), ("ds2", True)]:
        for t in range(30, 230):
            if not is_attack:
                regime = "clean_baseline"
                label = "Clean"
            elif t < 110:
                regime = "pre_attack_clean"
                label = "Clean"
            else:
                regime = "active_spoofing"
                label = "Spoofed"

            spoof_flag = 1.0 if label == "Spoofed" else 0.0
            row = {
                "scenario": scenario,
                "platform": "dynamic" if "Dynamic" in scenario else "static",
                "attack_type": "none" if not is_attack else scenario,
                "time_s": float(t),
                "num_channels": 7,
                "regime": regime,
                "is_transition": False,
                "label": label,
                "cn0_mean_dbhz": 48.5 + 1.5 * spoof_flag + rng.normal(0, 0.5),
                "cn0_std_dbhz": 1.2 + 0.3 * spoof_flag + abs(rng.normal(0, 0.2)),
                "cn0_temporal_jitter_dbhz": 0.3 + 0.4 * spoof_flag + abs(rng.normal(0, 0.1)),
                "sqm_ratio_mean": 0.76 + 0.055 * spoof_flag + rng.normal(0, 0.008),
                "sqm_ratio_max": 0.78 + 0.075 * spoof_flag + rng.normal(0, 0.010),
                "sqm_ratio_std": 0.01 + 0.015 * spoof_flag + abs(rng.normal(0, 0.003)),
                "sqm_delta_abs_mean": 0.028 + 0.012 * spoof_flag + abs(rng.normal(0, 0.004)),
                "sqm_delta_std": 0.01 + 0.01 * spoof_flag + abs(rng.normal(0, 0.002)),
                "corr_peak_norm_mean": 2.1 + 1.4 * spoof_flag + rng.normal(0, 0.2),
                "corr_peak_cv": 0.04 + 0.03 * spoof_flag + abs(rng.normal(0, 0.01)),
                "iq_quadrature_ratio": 0.08 + 0.03 * spoof_flag + abs(rng.normal(0, 0.01)),
                "doppler_residual_hz": 0.2 + 0.3 * spoof_flag + abs(rng.normal(0, 0.05)),
                "phase_slip_rate": 0.01 + 0.06 * spoof_flag + abs(rng.normal(0, 0.01)),
                "pll_lock_mean": 0.98 - 0.04 * spoof_flag + rng.normal(0, 0.005),
                "dll_code_error_rms_chips": 0.03 + 0.02 * spoof_flag + abs(rng.normal(0, 0.005)),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def test_temporal_block_split_enforces_guard_gap() -> None:
    df = _make_synthetic_feature_table()
    train_df, test_df = split_temporal_blocks(df, train_fraction=0.70, guard_gap_s=10.0)

    for (sc, reg), tr_grp in train_df.groupby(["scenario", "regime"]):
        te_grp = test_df[(test_df["scenario"] == sc) & (test_df["regime"] == reg)]
        assert float(te_grp["time_s"].min()) - float(tr_grp["time_s"].max()) >= 9.99


def test_stratified_and_temporal_pipelines_achieve_high_f1() -> None:
    df = _make_synthetic_feature_table()
    strat = evaluate_stratified_split(df, feature_cols=FEATURE_COLUMNS)
    temp = evaluate_temporal_block_split(df, feature_cols=FEATURE_COLUMNS, guard_gap_s=10.0)

    for model_name in ("SVM", "Random Forest", "MLP (Neural Net)"):
        assert strat[model_name].f1 > 0.95
        assert temp[model_name].f1 > 0.95
