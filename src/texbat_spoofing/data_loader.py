"""Data loading and multi-scenario feature extraction for TEXBAT GPS spoofing detection.

Supports three data tiers seamlessly:
1. Pre-extracted real TEXBAT 1-second multi-channel dataset (`data/processed/texbat_features_1s.csv`),
   shipped with the repository for instant reproducible ML benchmarking across CleanStatic,
   CleanDynamic, and DS1-DS6.
2. GNSS-SDR correlator & tracking `.mat` (HDF5 v7.3) dumps (`data/texbat_gnsssdr/<scenario>/`),
   enabling full regeneration of `texbat_features_1s.csv` from raw correlator outputs (`abs_E`,
   `abs_P`, `abs_L`, `Prompt_I`, `Prompt_Q`, `CN0_SNV_dB_Hz`, `carrier_doppler_hz`,
   `acc_carrier_phase_rad`, `carrier_lock_test`, `code_error_chips`).
3. Raw 25 MHz interleaved `int16` `.bin` RF captures (`data/raw/`), with automatic fallback
   to the physically calibrated synthetic 25 MHz RF generator when `.bin` files are not present.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from .signal_processing import DEFAULT_FS_HZ, load_iq_from_bin
from .synthetic_texbat import generate_synthetic_iq_scenario


@dataclass(frozen=True)
class TexbatScenarioSpec:
    """Metadata and ground-truth timing for an official UT Austin TEXBAT scenario."""

    name: str
    display_name: str
    platform: str  # "static" or "dynamic"
    attack_type: str  # "none", "switch_over", "overpowered_time", "matched_time", "matched_pos"
    power_advantage_db: float
    attack_onset_s: float | None  # None for purely clean baseline captures
    transition_guard_s: float = 8.0  # Guard band around onset excluded from clean/spoof training labels


# Ground-truth TEXBAT scenario specifications (Humphreys et al., ION GNSS 2012)
TEXBAT_SCENARIOS: dict[str, TexbatScenarioSpec] = {
    "cleanStatic": TexbatScenarioSpec(
        name="cleanStatic",
        display_name="Clean Static",
        platform="static",
        attack_type="none",
        power_advantage_db=0.0,
        attack_onset_s=None,
    ),
    "cleanDynamic": TexbatScenarioSpec(
        name="cleanDynamic",
        display_name="Clean Dynamic",
        platform="dynamic",
        attack_type="none",
        power_advantage_db=0.0,
        attack_onset_s=None,
    ),
    "ds1": TexbatScenarioSpec(
        name="ds1",
        display_name="DS1 (Static Switch-Over)",
        platform="static",
        attack_type="switch_over",
        power_advantage_db=0.0,
        attack_onset_s=125.0,
    ),
    "ds2": TexbatScenarioSpec(
        name="ds2",
        display_name="DS2 (Static Overpowered Time-Push +10 dB)",
        platform="static",
        attack_type="overpowered_time",
        power_advantage_db=10.0,
        attack_onset_s=110.0,
    ),
    "ds3": TexbatScenarioSpec(
        name="ds3",
        display_name="DS3 (Static Matched-Power Time-Push +1.3 dB)",
        platform="static",
        attack_type="matched_time",
        power_advantage_db=1.3,
        attack_onset_s=120.0,
    ),
    "ds4": TexbatScenarioSpec(
        name="ds4",
        display_name="DS4 (Static Matched-Power Pos-Push +0.4 dB)",
        platform="static",
        attack_type="matched_pos",
        power_advantage_db=0.4,
        attack_onset_s=115.0,
    ),
    "ds5": TexbatScenarioSpec(
        name="ds5",
        display_name="DS5 (Dynamic Overpowered Time-Push +9.9 dB)",
        platform="dynamic",
        attack_type="overpowered_time",
        power_advantage_db=9.9,
        attack_onset_s=105.0,
    ),
    "ds6": TexbatScenarioSpec(
        name="ds6",
        display_name="DS6 (Dynamic Matched-Power Pos-Push +0.8 dB)",
        platform="dynamic",
        attack_type="matched_pos",
        power_advantage_db=0.8,
        attack_onset_s=105.0,
    ),
}

# Feature columns used by the multi-channel TEXBAT ML pipeline
FEATURE_COLUMNS: list[str] = [
    "cn0_mean_dbhz",
    "cn0_std_dbhz",
    "cn0_temporal_jitter_dbhz",
    "sqm_ratio_mean",
    "sqm_ratio_max",
    "sqm_ratio_std",
    "sqm_delta_abs_mean",
    "sqm_delta_std",
    "corr_peak_norm_mean",
    "corr_peak_cv",
    "iq_quadrature_ratio",
    "doppler_residual_hz",
    "phase_slip_rate",
    "pll_lock_mean",
    "dll_code_error_rms_chips",
]


def load_or_synthesize_raw_iq(
    scenario_key: str,
    raw_dir: str | Path = "data/raw",
    num_samples: int = 100_000,
    start_time_s: float = 160.0,
    fs: float = DEFAULT_FS_HZ,
    seed: int = 42,
) -> tuple[np.ndarray, str]:
    """Load raw `.bin` I/Q if available in `raw_dir`, otherwise synthesize realistic 25 MHz I/Q.

    Returns `(iq_complex64, source_tag)` where `source_tag` is `'bin'` or `'synthetic'`.
    Note that `start_time_s` defaults to `160.0 s` for spoofing `.bin` captures so that
    analysis targets the active spoofing window rather than the `t=0` pre-attack baseline.
    """
    raw_path = Path(raw_dir)
    candidates = [
        raw_path / f"{scenario_key}.bin",
        raw_path / f"{scenario_key[0].upper() + scenario_key[1:]}.bin",
        raw_path / f"{scenario_key.lower()}.bin",
    ]
    for cand in candidates:
        if cand.exists():
            is_clean = scenario_key.lower().startswith("clean")
            t_offset = 5.0 if is_clean else start_time_s
            try:
                iq = load_iq_from_bin(cand, num_samples=num_samples, start_time_s=t_offset, fs=fs)
                return iq, "bin"
            except ValueError:
                iq = load_iq_from_bin(cand, num_samples=num_samples, start_sample=0, fs=fs)
                return iq, "bin"

    duration_ms = (num_samples / fs) * 1000.0
    iq = generate_synthetic_iq_scenario(scenario=scenario_key, duration_ms=duration_ms, fs=fs, seed=seed)
    return iq, "synthetic"


def extract_gnsssdr_scenario_windows(
    scenario_dir: str | Path,
    spec: TexbatScenarioSpec,
    fs: float = DEFAULT_FS_HZ,
    window_s: float = 1.0,
    warmup_s: float = 25.0,
) -> pd.DataFrame:
    """Extract 1-second multi-channel SQM and tracking features from GNSS-SDR `.mat` files."""
    import h5py

    sc_path = Path(scenario_dir)
    tracking_files = sorted(sc_path.glob("tracking_ch_*.mat"))
    if not tracking_files:
        raise FileNotFoundError(f"No `tracking_ch_*.mat` files found in {sc_path}")

    channel_frames: list[pd.DataFrame] = []
    for ch_idx, tf in enumerate(tracking_files):
        with h5py.File(tf, "r") as f:
            sample_count = f["PRN_start_sample_count"][()].ravel().astype(np.float64)
            if sample_count.size < 10:
                continue
            t_sec = sample_count / fs
            dt_ms = np.diff(t_sec, prepend=t_sec[0] - 0.020) * 1000.0
            dt_ms = np.clip(dt_ms, 0.5, 50.0)

            cn0 = f["CN0_SNV_dB_Hz"][()].ravel().astype(np.float64)
            abs_e = f["abs_E"][()].ravel().astype(np.float64)
            abs_p = np.maximum(f["abs_P"][()].ravel().astype(np.float64), 1.0)
            abs_l = f["abs_L"][()].ravel().astype(np.float64)
            prompt_i = f["Prompt_I"][()].ravel().astype(np.float64)
            prompt_q = f["Prompt_Q"][()].ravel().astype(np.float64)
            doppler = f["carrier_doppler_hz"][()].ravel().astype(np.float64)
            carr_err = f["carr_error_hz"][()].ravel().astype(np.float64)
            lock_test = f["carrier_lock_test"][()].ravel().astype(np.float64)
            code_err = f["code_error_chips"][()].ravel().astype(np.float64)
            prn = f["PRN"][()].ravel().astype(np.int32)

        # Keep valid post-bit-sync tracking epochs after cold-start warmup
        valid = (t_sec >= warmup_s) & (cn0 > 20.0) & (dt_ms >= 15.0)
        if not np.any(valid):
            valid = (t_sec >= warmup_s) & (cn0 > 20.0)
        if not np.any(valid):
            continue

        t_v = t_sec[valid]
        dt_v = dt_ms[valid]
        abs_p_v = abs_p[valid]
        abs_e_v = abs_e[valid]
        abs_l_v = abs_l[valid]

        sqm_ratio = (abs_e_v + abs_l_v) / (2.0 * abs_p_v)
        sqm_delta = (abs_e_v - abs_l_v) / (2.0 * abs_p_v)
        # Integration-rate-normalized prompt correlator magnitude (per ms, scaled by 1e5)
        corr_peak_norm = (abs_p_v / dt_v) / 1e5
        iq_quad_ratio = np.abs(prompt_q[valid]) / np.maximum(np.abs(prompt_i[valid]), 1.0)

        # Detrended Doppler residual within channel (removes smooth satellite orbital Doppler drift)
        dop_v = doppler[valid]
        dop_diff_hz = np.abs(np.diff(dop_v, prepend=dop_v[0]))
        phase_slip = ((lock_test[valid] < 0.65) | ( np.abs(carr_err[valid]) > 0.18 )).astype(np.float64)

        df_ch = pd.DataFrame(
            {
                "channel": ch_idx,
                "prn": prn[valid],
                "time_bin": np.floor(t_v / window_s).astype(np.int32),
                "cn0": cn0[valid],
                "sqm_ratio": sqm_ratio,
                "sqm_delta": sqm_delta,
                "sqm_delta_abs": np.abs(sqm_delta),
                "corr_peak_norm": corr_peak_norm,
                "iq_quad_ratio": iq_quad_ratio,
                "dop_step_hz": dop_diff_hz,
                "phase_slip": phase_slip,
                "pll_lock": lock_test[valid],
                "code_err_sq": code_err[valid] ** 2,
            }
        )
        # First aggregate per (channel, 1s window) so fast-rate channels don't dominate channel averages
        ch_1s = (
            df_ch.groupby(["time_bin", "channel"], as_index=False)
            .agg(
                prn=("prn", "first"),
                cn0_ch_mean=("cn0", "mean"),
                cn0_ch_std=("cn0", "std"),
                sqm_ratio_ch=("sqm_ratio", "mean"),
                sqm_delta_ch=("sqm_delta", "mean"),
                sqm_delta_abs_ch=("sqm_delta_abs", "mean"),
                corr_peak_ch=("corr_peak_norm", "mean"),
                corr_peak_ch_std=("corr_peak_norm", "std"),
                iq_quad_ch=("iq_quad_ratio", "mean"),
                dop_step_ch=("dop_step_hz", "mean"),
                phase_slip_ch=("phase_slip", "mean"),
                pll_lock_ch=("pll_lock", "mean"),
                code_err_ms_ch=("code_err_sq", "mean"),
            )
            .fillna(0.0)
        )
        channel_frames.append(ch_1s)

    if not channel_frames:
        raise RuntimeError(f"No valid post-warmup tracking epochs extracted from {sc_path}")

    all_ch = pd.concat(channel_frames, ignore_index=True)

    # Aggregate across all active satellite channels per 1-second receiver epoch
    grouped = all_ch.groupby("time_bin", as_index=False).agg(
        num_channels=("channel", "count"),
        cn0_mean_dbhz=("cn0_ch_mean", "mean"),
        cn0_std_dbhz=("cn0_ch_mean", "std"),
        cn0_temporal_jitter_dbhz=("cn0_ch_std", "mean"),
        sqm_ratio_mean=("sqm_ratio_ch", "mean"),
        sqm_ratio_max=("sqm_ratio_ch", "max"),
        sqm_ratio_std=("sqm_ratio_ch", "std"),
        sqm_delta_abs_mean=("sqm_delta_abs_ch", "mean"),
        sqm_delta_std=("sqm_delta_ch", "std"),
        corr_peak_norm_mean=("corr_peak_ch", "mean"),
        corr_peak_within_std=("corr_peak_ch_std", "mean"),
        iq_quadrature_ratio=("iq_quad_ch", "mean"),
        doppler_residual_hz=("dop_step_ch", "mean"),
        phase_slip_rate=("phase_slip_ch", "mean"),
        pll_lock_mean=("pll_lock_ch", "mean"),
        code_err_ms=("code_err_ms_ch", "mean"),
    ).fillna(0.0)

    # Require at least 4 tracked satellites for a valid receiver epoch
    grouped = grouped[grouped["num_channels"] >= 4].copy()
    grouped["time_s"] = grouped["time_bin"].astype(np.float64) * window_s
    grouped["corr_peak_cv"] = grouped["corr_peak_within_std"] / np.maximum(grouped["corr_peak_norm_mean"], 1e-6)
    grouped["dll_code_error_rms_chips"] = np.sqrt(np.maximum(grouped["code_err_ms"], 0.0))

    # Assign physically accurate ground-truth regime & binary label
    grouped["scenario"] = spec.name
    grouped["platform"] = spec.platform
    grouped["attack_type"] = spec.attack_type

    if spec.attack_onset_s is None:
        grouped["regime"] = "clean_baseline"
        grouped["label"] = "Clean"
        grouped["is_transition"] = False
    else:
        onset = spec.attack_onset_s
        guard = spec.transition_guard_s
        pre_mask = grouped["time_s"] < (onset - guard)
        post_mask = grouped["time_s"] >= (onset + guard)
        trans_mask = ~(pre_mask | post_mask)

        grouped["regime"] = np.where(
            pre_mask, "pre_attack_clean", np.where(post_mask, "active_spoofing", "transition_guard")
        )
        grouped["label"] = np.where(post_mask, "Spoofed", "Clean")
        grouped["is_transition"] = trans_mask

    cols_order = [
        "scenario",
        "platform",
        "attack_type",
        "time_s",
        "num_channels",
        "regime",
        "is_transition",
        "label",
    ] + FEATURE_COLUMNS

    return grouped[cols_order].reset_index(drop=True)


def build_texbat_gnsssdr_dataset(
    gnsssdr_root: str | Path = "data/texbat_gnsssdr",
    output_csv: str | Path | None = "data/processed/texbat_features_1s.csv",
    window_s: float = 1.0,
) -> pd.DataFrame:
    """Build the full multi-scenario 1-second TEXBAT dataset from GNSS-SDR `.mat` files."""
    root = Path(gnsssdr_root)
    frames: list[pd.DataFrame] = []
    for sc_key, spec in TEXBAT_SCENARIOS.items():
        sc_dir = root / sc_key
        if not sc_dir.exists():
            warnings.warn(f"Scenario directory missing, skipping: {sc_dir}", stacklevel=2)
            continue
        df_sc = extract_gnsssdr_scenario_windows(sc_dir, spec=spec, window_s=window_s)
        frames.append(df_sc)

    if not frames:
        raise FileNotFoundError(f"No TEXBAT scenario folders found under {root}")

    full_df = pd.concat(frames, ignore_index=True)
    if output_csv is not None:
        out_path = Path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        full_df.to_csv(out_path, index=False)
    return full_df


def load_texbat_feature_dataset(
    processed_csv: str | Path = "data/processed/texbat_features_1s.csv",
    gnsssdr_root: str | Path = "data/texbat_gnsssdr",
    exclude_transition: bool = True,
) -> pd.DataFrame:
    """Load the 1-second TEXBAT feature dataset from CSV (or build from `.mat` if needed)."""
    csv_path = Path(processed_csv)
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    elif Path(gnsssdr_root).exists():
        df = build_texbat_gnsssdr_dataset(gnsssdr_root=gnsssdr_root, output_csv=csv_path)
    else:
        raise FileNotFoundError(
            f"Neither pre-extracted dataset {csv_path} nor GNSS-SDR directory {gnsssdr_root} exists."
        )

    if exclude_transition and "is_transition" in df.columns:
        df = df[~df["is_transition"].astype(bool)].reset_index(drop=True)
    return df
