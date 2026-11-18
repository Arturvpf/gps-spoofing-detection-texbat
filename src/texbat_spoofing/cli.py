"""Command-line interface for TEXBAT GPS spoofing feature extraction and ML evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .classifiers import (
    evaluate_leave_one_attack_out,
    evaluate_stratified_split,
    evaluate_temporal_block_split,
)
from .data_loader import (
    FEATURE_COLUMNS,
    build_texbat_gnsssdr_dataset,
    load_or_synthesize_raw_iq,
    load_texbat_feature_dataset,
)
from .sqm_features import extract_multichannel_iq_features


def run_full_benchmark(
    processed_csv: Path,
    gnsssdr_root: Path,
    summary_json: Path,
    rebuild_features: bool = False,
) -> dict:
    if rebuild_features and gnsssdr_root.exists():
        print(f"[1/4] Extracting 1s multi-channel SQM features from {gnsssdr_root} ...")
        build_texbat_gnsssdr_dataset(gnsssdr_root=gnsssdr_root, output_csv=processed_csv)

    df = load_texbat_feature_dataset(processed_csv=processed_csv, gnsssdr_root=gnsssdr_root, exclude_transition=True)
    print(f"Loaded dataset: {len(df)} 1s epochs across {df['scenario'].nunique()} TEXBAT scenarios.")
    print(df.groupby(["scenario", "label"]).size().unstack(fill_value=0).to_string())

    print("\n[2/4] Evaluating Stratified 70/30 Split + 5-Fold CV ...")
    strat_evals = evaluate_stratified_split(df, feature_cols=FEATURE_COLUMNS)
    for name, ev in strat_evals.items():
        print(
            f"  {name:18s} | Acc={ev.accuracy:.4f} | Prec={ev.precision:.4f} | "
            f"Rec={ev.recall:.4f} | F1={ev.f1:.4f} | ROC-AUC={ev.roc_auc:.4f} | params={ev.best_params}"
        )

    print("\n[3/4] Evaluating Leakage-Free Temporal Block Split (10s guard gap) ...")
    temp_evals = evaluate_temporal_block_split(df, feature_cols=FEATURE_COLUMNS, guard_gap_s=10.0)
    for name, ev in temp_evals.items():
        print(
            f"  {name:18s} | Acc={ev.accuracy:.4f} | Prec={ev.precision:.4f} | "
            f"Rec={ev.recall:.4f} | F1={ev.f1:.4f} | ROC-AUC={ev.roc_auc:.4f}"
        )

    print("\n[4/4] Evaluating Leave-One-Attack-Out (LOAO) Generalization across DS1-DS6 ...")
    loao_df = evaluate_leave_one_attack_out(df, feature_cols=FEATURE_COLUMNS)
    loao_summary = (
        loao_df.groupby("model", as_index=False)[["unseen_attack_recall", "clean_false_alarm_rate", "fold_f1", "fold_roc_auc"]]
        .mean()
    )
    print(loao_summary.to_string(index=False))

    summary = {
        "dataset": {
            "total_epochs": int(len(df)),
            "clean_epochs": int((df["label"] == "Clean").sum()),
            "spoofed_epochs": int((df["label"] == "Spoofed").sum()),
            "scenarios": sorted(df["scenario"].unique().tolist()),
            "features": FEATURE_COLUMNS,
        },
        "stratified_70_30": {
            k: {
                "accuracy": round(v.accuracy, 4),
                "precision": round(v.precision, 4),
                "recall": round(v.recall, 4),
                "f1": round(v.f1, 4),
                "roc_auc": round(v.roc_auc, 4),
                "pr_auc": round(v.pr_auc, 4),
                "cv_mean_f1": round(v.cv_mean_f1, 4),
                "best_params": v.best_params,
                "confusion_matrix": v.confusion_mat.tolist(),
            }
            for k, v in strat_evals.items()
        },
        "temporal_block_70_30": {
            k: {
                "accuracy": round(v.accuracy, 4),
                "precision": round(v.precision, 4),
                "recall": round(v.recall, 4),
                "f1": round(v.f1, 4),
                "roc_auc": round(v.roc_auc, 4),
                "pr_auc": round(v.pr_auc, 4),
                "best_params": v.best_params,
                "confusion_matrix": v.confusion_mat.tolist(),
            }
            for k, v in temp_evals.items()
        },
        "leave_one_attack_out": loao_df.to_dict(orient="records"),
    }

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved full evaluation summary to {summary_json}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TEXBAT GPS Spoofing Detection Pipeline")
    parser.add_argument(
        "--processed-csv",
        type=Path,
        default=Path("data/processed/texbat_features_1s.csv"),
        help="Path to 1-second extracted TEXBAT feature CSV.",
    )
    parser.add_argument(
        "--gnsssdr-root",
        type=Path,
        default=Path("data/texbat_gnsssdr"),
        help="Root folder containing per-scenario GNSS-SDR .mat files.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("data/processed/evaluation_summary.json"),
        help="Output path for JSON evaluation summary.",
    )
    parser.add_argument(
        "--rebuild-features",
        action="store_true",
        help="Re-extract features from GNSS-SDR .mat files if available.",
    )
    parser.add_argument(
        "--demo-iq",
        action="store_true",
        help="Run a quick 25 MHz raw/synthetic I/Q 2D acquisition + SQM waveform demo.",
    )
    args = parser.parse_args(argv)

    if args.demo_iq:
        print("=== 25 MHz Baseband I/Q Acquisition & SQM Demo ===")
        for sc in ["cleanStatic", "cleanDynamic", "ds1", "ds2", "ds3"]:
            iq, src = load_or_synthesize_raw_iq(sc, num_samples=50_000)
            feats = extract_multichannel_iq_features(iq)
            print(
                f"  {sc:14s} ({src:9s}) | PRN={int(feats['PRN']):2d} | "
                f"C/N0={feats['C/N0 (dB-Hz)']:5.2f} dB-Hz | HPW={feats['Largura HPW (chips)']:.3f} ch | "
                f"Skew={feats['Assimetria (Skew)']:+.3f} | RCP={feats['RCP (P/S)']:5.2f}"
            )

    run_full_benchmark(
        processed_csv=args.processed_csv,
        gnsssdr_root=args.gnsssdr_root,
        summary_json=args.summary_json,
        rebuild_features=args.rebuild_features,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
