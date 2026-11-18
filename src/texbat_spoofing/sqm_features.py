"""Signal Quality Monitoring (SQM) and multi-channel feature extraction for GPS spoofing detection.

Provides:
1. Full-waveform correlation-peak SQM metrics (for 25 MHz raw I/Q `.bin` or synthetic signals):
   - Ratio of the Correlation Peak (RCP): main peak to strongest secondary peak outside +-2.0 chips.
   - Sub-sample interpolated Half-Peak Width (HPW, chips) above noise baseline.
   - Sub-sample Leading/Trailing Asymmetry (Skew).
   - Early-Prompt-Late Ratio, Delta (Asymmetry), and Double-Delta SQM metrics.
2. Multi-channel 1-second receiver epoch feature extraction from GNSS-SDR tracking/observables
   (`.mat` v7.3 HDF5 files) across TEXBAT scenarios (CleanStatic, CleanDynamic, DS1-DS6).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .signal_processing import (
    AcquisitionResult,
    DEFAULT_FS_HZ,
    GPS_CHIP_RATE_HZ,
    SpectralMetrics,
    acquire_prn_pcps,
    circular_exclusion_mask,
    compute_psd_and_flatness,
    scan_visible_satellites,
)


@dataclass(frozen=True)
class RCPResult:
    """Primary vs. secondary correlation peak analysis."""

    rcp: float
    primary_peak_idx: int
    primary_peak_val: float
    secondary_peak_idx: int
    secondary_peak_val: float
    exclusion_radius_samples: int


def calculate_rcp(
    corr_profile_norm: np.ndarray,
    phase_index: int | None = None,
    fs: float = DEFAULT_FS_HZ,
    exclusion_chips: float = 2.0,
) -> RCPResult:
    """Compute Ratio of the Correlation Peak (RCP = primary_peak / secondary_peak).

    Excludes `[phase_index - exclusion_chips, phase_index + exclusion_chips]` with
    proper circular wrap-around across the 1 ms C/A code period.
    """
    corr = np.asarray(corr_profile_norm, dtype=np.float64)
    n_samples = corr.size
    if n_samples == 0:
        raise ValueError("`corr_profile_norm` must not be empty.")

    if phase_index is None:
        phase_index = int(np.argmax(corr))
    else:
        phase_index = int(phase_index) % n_samples

    samples_per_chip = fs / GPS_CHIP_RATE_HZ
    excl_radius = int(np.ceil(exclusion_chips * samples_per_chip))
    outside_mask = circular_exclusion_mask(n_samples, phase_index, excl_radius)

    primary_val = float(corr[phase_index])
    if not np.any(outside_mask):
        return RCPResult(
            rcp=999.0,
            primary_peak_idx=phase_index,
            primary_peak_val=primary_val,
            secondary_peak_idx=phase_index,
            secondary_peak_val=0.0,
            exclusion_radius_samples=excl_radius,
        )

    masked_corr = np.where(outside_mask, corr, -np.inf)
    sec_idx = int(np.argmax(masked_corr))
    sec_val = float(corr[sec_idx])
    rcp = primary_val / sec_val if sec_val > 1e-10 else 999.0

    return RCPResult(
        rcp=float(rcp),
        primary_peak_idx=phase_index,
        primary_peak_val=primary_val,
        secondary_peak_idx=sec_idx,
        secondary_peak_val=sec_val,
        exclusion_radius_samples=excl_radius,
    )


def _circular_interp(corr: np.ndarray, fractional_idx: float) -> float:
    """Linearly interpolate circular correlation array at `fractional_idx`."""
    n = corr.size
    x = fractional_idx % n
    i0 = int(np.floor(x)) % n
    i1 = (i0 + 1) % n
    alpha = x - np.floor(x)
    return float((1.0 - alpha) * corr[i0] + alpha * corr[i1])


def _subsample_half_height_widths(
    corr: np.ndarray,
    peak_idx: int,
    baseline: float,
    max_search_samples: int,
) -> tuple[float, float]:
    """Return sub-sample leading and trailing half-excess-height widths around `peak_idx`."""
    n = corr.size
    peak_val = float(corr[peak_idx])
    target = baseline + 0.5 * max(peak_val - baseline, 1e-9)

    # Search left (leading edge: peak_idx - k)
    lead_width = 0.0
    prev_val = peak_val
    for k in range(1, max_search_samples + 1):
        cur_val = float(corr[(peak_idx - k) % n])
        if cur_val <= target:
            denom = max(prev_val - cur_val, 1e-12)
            frac = (prev_val - target) / denom
            lead_width = (k - 1) + frac
            break
        prev_val = cur_val
    else:
        lead_width = float(max_search_samples)

    # Search right (trailing edge: peak_idx + k)
    trail_width = 0.0
    prev_val = peak_val
    for k in range(1, max_search_samples + 1):
        cur_val = float(corr[(peak_idx + k) % n])
        if cur_val <= target:
            denom = max(prev_val - cur_val, 1e-12)
            frac = (prev_val - target) / denom
            trail_width = (k - 1) + frac
            break
        prev_val = cur_val
    else:
        trail_width = float(max_search_samples)

    return lead_width, trail_width


def extract_waveform_sqm_features(
    acq: AcquisitionResult,
    fs: float = DEFAULT_FS_HZ,
    spectral: SpectralMetrics | None = None,
    correlator_spacing_chips: float = 0.5,
) -> dict[str, float]:
    """Extract SQM and correlation-peak shape features from a 2D PCPS `AcquisitionResult`.

    For an ideal undistorted BPSK C/A triangular correlation peak above noise baseline:
    - `Largura HPW (chips)` = 1.000 chip (50% excess amplitude full width).
    - `Assimetria (Skew)` = 0.000.
    - `SQM Ratio (E+L)/2P` (at d=0.5 chips excess) ~= 0.50 (or ~0.54 including noise floor).
    - `SQM Delta (E-L)/2P` = 0.000.
    """
    corr = np.asarray(acq.corr_profile_norm, dtype=np.float64)
    n_samples = corr.size
    samples_per_chip = fs / GPS_CHIP_RATE_HZ
    peak_idx = int(acq.code_phase_samples) % n_samples

    rcp_res = calculate_rcp(corr, phase_index=peak_idx, fs=fs, exclusion_chips=2.0)

    # Noise baseline in normalized units is ~1.0 (since corr_profile_norm = corr / mean_noise)
    excl_mask = circular_exclusion_mask(n_samples, peak_idx, rcp_res.exclusion_radius_samples)
    baseline = float(np.median(corr[excl_mask])) if np.any(excl_mask) else 1.0

    max_search = max(4, int(np.ceil(2.5 * samples_per_chip)))
    lead_samples, trail_samples = _subsample_half_height_widths(
        corr=corr,
        peak_idx=peak_idx,
        baseline=baseline,
        max_search_samples=max_search,
    )
    total_width_samples = lead_samples + trail_samples
    hpw_chips = total_width_samples / samples_per_chip
    asymmetry = (lead_samples - trail_samples) / max(total_width_samples, 1e-9)

    # Early-Prompt-Late correlator taps (wide d = 0.5 chips and narrow d2 = 0.25 chips)
    d1_samples = correlator_spacing_chips * samples_per_chip
    d2_samples = 0.5 * d1_samples

    r_p = max(float(corr[peak_idx]), 1e-9)
    r_e1 = _circular_interp(corr, peak_idx - d1_samples)
    r_l1 = _circular_interp(corr, peak_idx + d1_samples)
    r_e2 = _circular_interp(corr, peak_idx - d2_samples)
    r_l2 = _circular_interp(corr, peak_idx + d2_samples)

    sqm_ratio = (r_e1 + r_l1) / (2.0 * r_p)
    sqm_delta = (r_e1 - r_l1) / (2.0 * r_p)
    sqm_double_delta = ((r_e1 - r_l1) - 2.0 * (r_e2 - r_l2)) / r_p

    out: dict[str, float] = {
        "PRN": float(acq.prn),
        "C/N0 (dB-Hz)": round(float(acq.cn0_dbhz), 3),
        "Pico SNR (P/R)": round(float(acq.peak_to_mean_snr), 3),
        "Doppler Encontrado (Hz)": round(float(acq.doppler_hz), 2),
        "Largura HPW (chips)": round(float(hpw_chips), 4),
        "Assimetria (Skew)": round(float(asymmetry), 4),
        "RCP (P/S)": round(float(rcp_res.rcp), 3),
        "SQM Ratio (E+L)/2P": round(float(sqm_ratio), 4),
        "SQM Delta (E-L)/2P": round(float(sqm_delta), 4),
        "SQM Double-Delta": round(float(sqm_double_delta), 4),
    }

    if spectral is not None:
        out["Spectral Flatness"] = round(float(spectral.spectral_flatness), 5)
        out["Wideband Power (dB)"] = round(float(spectral.wideband_power_db), 3)

    return out


def extract_multichannel_iq_features(
    iq_segment: np.ndarray,
    fs: float = DEFAULT_FS_HZ,
    acquisition_threshold: float = 4.6,
    top_k_channels: int = 6,
) -> dict[str, float]:
    """Extract both single-strongest-PRN and cross-channel SQM features from a raw I/Q segment."""
    spec = compute_psd_and_flatness(iq_segment, fs=fs)
    all_sats = scan_visible_satellites(
        iq_segment,
        fs=fs,
        doppler_range_hz=5000.0,
        doppler_step_hz=250.0,
        noncoherent_ms=1,
        acquisition_threshold=acquisition_threshold,
    )
    acquired = [r for r in all_sats if r.acquired][:top_k_channels]
    if not acquired:
        acquired = [all_sats[0]]

    # Refine Doppler (100 Hz step) on strongest satellite
    best_coarse = acquired[0]
    best_fine = acquire_prn_pcps(
        iq_segment,
        prn=best_coarse.prn,
        fs=fs,
        doppler_range_hz=5000.0,
        doppler_step_hz=100.0,
        noncoherent_ms=2 if iq_segment.size >= int(2 * fs * 1e-3) else 1,
    )
    base_feats = extract_waveform_sqm_features(best_fine, fs=fs, spectral=spec)

    cn0_vals = np.array([r.cn0_dbhz for r in acquired], dtype=np.float64)
    rcp_vals = np.array([calculate_rcp(r.corr_profile_norm, r.code_phase_samples, fs=fs).rcp for r in acquired], dtype=np.float64)

    base_feats["Visible Satellites"] = float(len(acquired))
    base_feats["C/N0 Mean Channels (dB-Hz)"] = round(float(np.mean(cn0_vals)), 3)
    base_feats["C/N0 Std Channels (dB-Hz)"] = round(float(np.std(cn0_vals)), 3)
    base_feats["RCP Min Channels"] = round(float(np.min(rcp_vals)), 3)
    return base_feats
