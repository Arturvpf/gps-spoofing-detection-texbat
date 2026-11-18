"""Unit tests for Signal Quality Monitoring (SQM) metrics on clean vs. spoofed correlation peaks."""

from __future__ import annotations

import numpy as np

from texbat_spoofing.signal_processing import acquire_prn_pcps
from texbat_spoofing.sqm_features import calculate_rcp, extract_waveform_sqm_features
from texbat_spoofing.synthetic_texbat import generate_synthetic_iq_scenario


def test_ideal_triangle_correlation_hpw_and_symmetry() -> None:
    """An undistorted triangular autocorrelation peak of +-1 chip has HPW = 1.0 chip and Skew = 0.0."""
    fs = 25_000_000.0
    n_1ms = 25_000
    samples_per_chip = fs / 1_023_000.0
    center = 5000
    dist_chips = np.abs(np.arange(n_1ms) - center) / samples_per_chip
    corr = 1.0 + 11.0 * np.maximum(0.0, 1.0 - dist_chips)

    from texbat_spoofing.signal_processing import AcquisitionResult

    acq = AcquisitionResult(
        prn=23,
        acquired=True,
        doppler_hz=-300.0,
        code_phase_samples=center,
        code_phase_chips=center / samples_per_chip,
        peak_to_mean_snr=12.0,
        coherent_power_snr=112.0,
        cn0_dbhz=50.5,
        corr_profile_norm=corr.astype(np.float32),
    )
    feats = extract_waveform_sqm_features(acq, fs=fs)
    assert abs(feats["Largura HPW (chips)"] - 1.0) < 0.02
    assert abs(feats["Assimetria (Skew)"]) < 1e-4
    assert abs(feats["SQM Delta (E-L)/2P"]) < 1e-4
    assert feats["RCP (P/S)"] > 10.0


def test_rcp_detects_secondary_spoofing_peak_near_wraparound() -> None:
    fs = 25_000_000.0
    corr = np.ones(25_000, dtype=np.float64)
    corr[10] = 12.0  # Main peak near index 0 boundary
    corr[24_000] = 6.0  # Secondary spoofing peak outside +-2 chips

    res = calculate_rcp(corr, phase_index=10, fs=fs, exclusion_chips=2.0)
    assert abs(res.rcp - 2.0) < 1e-6
    assert res.secondary_peak_idx == 24_000


def test_matched_power_spoofing_widens_hpw_and_distorts_sqm() -> None:
    fs = 10_230_000.0  # 10 samples/chip
    iq_clean = generate_synthetic_iq_scenario("CleanStatic", duration_ms=3.0, fs=fs, seed=11)
    iq_spoof = generate_synthetic_iq_scenario("DS3_MatchedPowerTimePush", duration_ms=3.0, fs=fs, seed=11)

    acq_clean = acquire_prn_pcps(iq_clean, prn=23, fs=fs, doppler_range_hz=1000.0, doppler_step_hz=100.0, noncoherent_ms=3)
    acq_spoof = acquire_prn_pcps(iq_spoof, prn=23, fs=fs, doppler_range_hz=1000.0, doppler_step_hz=100.0, noncoherent_ms=3)

    f_clean = extract_waveform_sqm_features(acq_clean, fs=fs)
    f_spoof = extract_waveform_sqm_features(acq_spoof, fs=fs)

    # Matched-power pull-off widens the correlation peak and increases E-L ratio
    assert f_spoof["Largura HPW (chips)"] > f_clean["Largura HPW (chips)"] + 0.15
    assert f_spoof["SQM Ratio (E+L)/2P"] > f_clean["SQM Ratio (E+L)/2P"] + 0.05
