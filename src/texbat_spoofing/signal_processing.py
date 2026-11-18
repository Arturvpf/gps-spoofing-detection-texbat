"""Signal-processing primitives for GPS L1 C/A acquisition and spectral analysis.

Implements:
- Interleaved int16 I/Q memory-mapped ingestion with time-offset support (`.bin`).
- Exact GPS ICD-GPS-200 Gold C/A code generation (PRNs 1..32) and zero-order-hold
  rectangular BPSK sampling (eliminating FFT `scipy.signal.resample` Gibbs ringing
  and multi-millisecond code-stretching bugs).
- Two-sided Welch Power Spectral Density (PSD), Wiener spectral flatness, and
  wideband I/Q power estimation.
- 2D Parallel Code-Phase Search (PCPS) FFT acquisition with non-coherent integration,
  circular peak-exclusion noise floor estimation, and Rayleigh-calibrated C/N0 (dB-Hz).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.signal import welch

GPS_CHIP_RATE_HZ: float = 1_023_000.0
GPS_CA_CODE_LENGTH: int = 1023
GPS_CA_PERIOD_S: float = 1e-3
DEFAULT_FS_HZ: float = 25_000_000.0

# Phase-select tap assignments on register G2 per IS-GPS-200 (Table 3-Ia).
G2_PHASE_TAPS: dict[int, tuple[int, int]] = {
    1: (2, 6), 2: (3, 7), 3: (4, 8), 4: (5, 9),
    5: (1, 9), 6: (2, 10), 7: (1, 8), 8: (2, 9),
    9: (3, 10), 10: (2, 3), 11: (3, 4), 12: (5, 6),
    13: (6, 7), 14: (7, 8), 15: (8, 9), 16: (9, 10),
    17: (1, 4), 18: (2, 5), 19: (3, 6), 20: (4, 7),
    21: (5, 8), 22: (6, 9), 23: (1, 3), 24: (4, 6),
    25: (5, 7), 26: (6, 8), 27: (7, 9), 28: (8, 10),
    29: (1, 6), 30: (2, 7), 31: (3, 8), 32: (4, 9),
}


@dataclass(frozen=True)
class AcquisitionResult:
    """Result of 2D PCPS acquisition for a single GPS PRN."""

    prn: int
    acquired: bool
    doppler_hz: float
    code_phase_samples: int
    code_phase_chips: float
    peak_to_mean_snr: float
    coherent_power_snr: float
    cn0_dbhz: float
    corr_profile_norm: np.ndarray
    doppler_grid_hz: np.ndarray | None = None
    acquisition_surface: np.ndarray | None = None


@dataclass(frozen=True)
class SpectralMetrics:
    """Frequency-domain and wideband power descriptors of a complex I/Q segment."""

    freqs_mhz: np.ndarray
    psd_db_per_hz: np.ndarray
    spectral_flatness: float
    wideband_power_db: float
    iq_imbalance_ratio: float


@lru_cache(maxsize=64)
def _generate_ca_code_cached(prn: int) -> tuple[int, ...]:
    if prn not in G2_PHASE_TAPS:
        raise ValueError(f"Invalid GPS PRN {prn}; expected integer in 1..32.")

    g1 = np.ones(10, dtype=np.int8)
    g2 = np.ones(10, dtype=np.int8)
    ca_bits = np.empty(GPS_CA_CODE_LENGTH, dtype=np.int8)

    tap_a, tap_b = G2_PHASE_TAPS[prn]
    idx_a, idx_b = tap_a - 1, tap_b - 1

    for i in range(GPS_CA_CODE_LENGTH):
        g1_out = g1[9]
        g2_out = g2[idx_a] ^ g2[idx_b]
        ca_bits[i] = g1_out ^ g2_out

        g1_fb = g1[2] ^ g1[9]
        g2_fb = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]

        g1[1:] = g1[:-1]
        g1[0] = g1_fb
        g2[1:] = g2[:-1]
        g2[0] = g2_fb

    # BPSK mapping: bit 0 -> +1, bit 1 -> -1
    bipolar = 1 - 2 * ca_bits
    return tuple(int(v) for v in bipolar)


def generate_ca_code(prn: int) -> np.ndarray:
    """Return the 1023-chip bipolar (+1 / -1) C/A Gold code for `prn` (1..32)."""
    return np.array(_generate_ca_code_cached(prn), dtype=np.float32)


def sample_ca_code(prn: int, fs: float = DEFAULT_FS_HZ, duration_ms: float = 1.0) -> np.ndarray:
    """Sample the periodic 1.023 Mchip/s rectangular C/A code at sampling rate `fs`.

    Unlike `scipy.signal.resample`, zero-order chip indexing preserves crisp BPSK
    rectangular transitions (no Gibbs ringing) and properly repeats the 1 ms C/A
    sequence when `duration_ms > 1.0`.
    """
    if fs <= 0 or duration_ms <= 0:
        raise ValueError("Sampling rate `fs` and `duration_ms` must be positive.")
    n_samples = int(round(fs * duration_ms * 1e-3))
    chips = generate_ca_code(prn)
    chip_indices = (np.arange(n_samples, dtype=np.float64) * (GPS_CHIP_RATE_HZ / fs)).astype(np.int64) % GPS_CA_CODE_LENGTH
    return chips[chip_indices]


@lru_cache(maxsize=128)
def _cached_ca_fft_conj(prn: int, fs: float) -> np.ndarray:
    ca_1ms = sample_ca_code(prn, fs=fs, duration_ms=1.0).astype(np.complex64)
    return np.conj(np.fft.fft(ca_1ms))


def load_iq_from_bin(
    path: str | Path,
    num_samples: int = 5_000_000,
    start_sample: int = 0,
    start_time_s: float | None = None,
    fs: float = DEFAULT_FS_HZ,
    remove_dc: bool = True,
) -> np.ndarray:
    """Memory-map a TEXBAT `.bin` file (interleaved `int16` I/Q) into `complex64`.

    Note on TEXBAT recordings: spoofing scenarios (`ds1`..`ds8`) begin with ~100-120 s
    of clean signal so receivers can lock onto authentic satellites first. Pass
    `start_time_s >= 130.0` to inspect active spoofing segments.
    """
    bin_path = Path(path)
    if not bin_path.exists():
        raise FileNotFoundError(f"TEXBAT binary recording not found: {bin_path}")

    if start_time_s is not None:
        if start_time_s < 0:
            raise ValueError("`start_time_s` must be non-negative.")
        start_sample = int(round(start_time_s * fs))

    if start_sample < 0 or num_samples <= 0:
        raise ValueError("`start_sample` must be >= 0 and `num_samples` must be > 0.")

    mm = np.memmap(bin_path, dtype=np.int16, mode="r")
    total_complex_samples = mm.size // 2
    if start_sample >= total_complex_samples:
        raise ValueError(
            f"Requested start_sample={start_sample} exceeds file length ({total_complex_samples} complex samples)."
        )

    end_sample = min(total_complex_samples, start_sample + num_samples)
    raw_interleaved = np.asarray(mm[start_sample * 2 : end_sample * 2], dtype=np.float32)
    iq = raw_interleaved[0::2] + 1j * raw_interleaved[1::2]
    iq = iq.astype(np.complex64, copy=False)

    if remove_dc and iq.size > 0:
        iq = iq - np.mean(iq)
    return iq


def compute_psd_and_flatness(
    iq_signal: np.ndarray,
    fs: float = DEFAULT_FS_HZ,
    nperseg: int = 1024,
    max_samples: int = 250_000,
) -> SpectralMetrics:
    """Compute two-sided Welch PSD, Wiener spectral flatness, and wideband power."""
    seg = np.asarray(iq_signal[:max_samples], dtype=np.complex64)
    if seg.size == 0:
        raise ValueError("Cannot compute PSD of an empty signal.")

    actual_nperseg = min(nperseg, seg.size)
    freqs, pxx = welch(
        seg,
        fs=fs,
        nperseg=actual_nperseg,
        return_onesided=False,
        scaling="density",
    )
    order = np.argsort(freqs)
    freqs = freqs[order]
    pxx = np.maximum(np.real(pxx[order]), 1e-30)

    geom_mean = float(np.exp(np.mean(np.log(pxx))))
    arith_mean = float(np.mean(pxx))
    spectral_flatness = float(np.clip(geom_mean / max(arith_mean, 1e-30), 0.0, 1.0))

    mean_power = float(np.mean(np.abs(seg) ** 2))
    wideband_power_db = 10.0 * np.log10(max(mean_power, 1e-30))

    std_i = float(np.std(np.real(seg)))
    std_q = float(np.std(np.imag(seg)))
    iq_imbalance = std_i / max(std_q, 1e-12)

    return SpectralMetrics(
        freqs_mhz=freqs / 1e6,
        psd_db_per_hz=10.0 * np.log10(pxx),
        spectral_flatness=spectral_flatness,
        wideband_power_db=wideband_power_db,
        iq_imbalance_ratio=iq_imbalance,
    )


def circular_exclusion_mask(n_samples: int, center_idx: int, exclusion_radius: int) -> np.ndarray:
    """Boolean mask that is True OUTSIDE `[center_idx - radius, center_idx + radius]` mod `n_samples`."""
    idx = np.arange(n_samples)
    circular_dist = np.minimum((idx - center_idx) % n_samples, (center_idx - idx) % n_samples)
    return circular_dist > exclusion_radius


def acquire_prn_pcps(
    iq_signal: np.ndarray,
    prn: int,
    fs: float = DEFAULT_FS_HZ,
    doppler_range_hz: float = 5000.0,
    doppler_step_hz: float = 100.0,
    noncoherent_ms: int = 1,
    acquisition_threshold: float = 4.5,
    keep_surface: bool = False,
) -> AcquisitionResult:
    """Run 2D FFT Parallel Code-Phase Search (PCPS) acquisition for a single PRN."""
    n_1ms = int(round(fs * GPS_CA_PERIOD_S))
    if iq_signal.size < n_1ms:
        raise ValueError(f"Signal length ({iq_signal.size}) is shorter than 1 ms ({n_1ms} samples at {fs} Hz).")

    n_blocks = max(1, min(noncoherent_ms, iq_signal.size // n_1ms))
    doppler_bins = np.arange(-doppler_range_hz, doppler_range_hz + 0.5 * doppler_step_hz, doppler_step_hz, dtype=np.float64)
    ca_fft_conj = _cached_ca_fft_conj(prn, float(fs))

    t_1ms = np.arange(n_1ms, dtype=np.float64) / fs
    surface = np.zeros((doppler_bins.size, n_1ms), dtype=np.float64)
    coherent_power_surface = np.zeros((doppler_bins.size, n_1ms), dtype=np.float64)

    for blk in range(n_blocks):
        seg = np.asarray(iq_signal[blk * n_1ms : (blk + 1) * n_1ms], dtype=np.complex64)
        for d_idx, f_d in enumerate(doppler_bins):
            wiped = seg * np.exp(-1j * 2.0 * np.pi * f_d * t_1ms).astype(np.complex64)
            corr = np.fft.ifft(np.fft.fft(wiped) * ca_fft_conj)
            mag = np.abs(corr)
            surface[d_idx, :] += mag
            coherent_power_surface[d_idx, :] += mag ** 2

    surface /= n_blocks
    coherent_power_surface /= n_blocks

    best_d_idx, best_phase_idx = np.unravel_index(np.argmax(surface), surface.shape)
    found_doppler = float(doppler_bins[best_d_idx])
    corr_row = surface[best_d_idx, :]
    power_row = coherent_power_surface[best_d_idx, :]

    samples_per_chip = fs / GPS_CHIP_RATE_HZ
    excl_radius = int(np.ceil(2.0 * samples_per_chip))
    noise_mask = circular_exclusion_mask(n_1ms, int(best_phase_idx), excl_radius)

    mean_noise_mag = float(np.mean(corr_row[noise_mask])) if np.any(noise_mask) else 1.0
    mean_noise_mag = max(mean_noise_mag, 1e-12)
    corr_norm = corr_row / mean_noise_mag
    peak_to_mean_snr = float(corr_norm[best_phase_idx])

    # Calibrated post-correlation power SNR -> C/N0 (dB-Hz) for 1 ms coherent integration:
    # E[|R_noise|^2] = 2*sigma_n^2, E[|R_peak|^2] = A^2 + 2*sigma_n^2 => SNR_coh = (P_peak - P_noise)/P_noise
    mean_noise_pow = float(np.mean(power_row[noise_mask])) if np.any(noise_mask) else 1.0
    mean_noise_pow = max(mean_noise_pow, 1e-20)
    peak_pow = float(power_row[best_phase_idx])
    coherent_snr = max((peak_pow - mean_noise_pow) / mean_noise_pow, 1e-3)
    cn0_dbhz = float(10.0 * np.log10(coherent_snr / GPS_CA_PERIOD_S))

    code_phase_chips = float(best_phase_idx / samples_per_chip)

    return AcquisitionResult(
        prn=int(prn),
        acquired=bool(peak_to_mean_snr >= acquisition_threshold),
        doppler_hz=found_doppler,
        code_phase_samples=int(best_phase_idx),
        code_phase_chips=code_phase_chips,
        peak_to_mean_snr=peak_to_mean_snr,
        coherent_power_snr=float(coherent_snr),
        cn0_dbhz=cn0_dbhz,
        corr_profile_norm=corr_norm.astype(np.float32),
        doppler_grid_hz=doppler_bins if keep_surface else None,
        acquisition_surface=surface.astype(np.float32) if keep_surface else None,
    )


def scan_visible_satellites(
    iq_signal: np.ndarray,
    fs: float = DEFAULT_FS_HZ,
    prns: Iterable[int] = range(1, 33),
    doppler_range_hz: float = 5000.0,
    doppler_step_hz: float = 250.0,
    noncoherent_ms: int = 1,
    acquisition_threshold: float = 4.8,
) -> list[AcquisitionResult]:
    """Acquire across candidate PRNs and return results sorted by descending SNR."""
    results = [
        acquire_prn_pcps(
            iq_signal=iq_signal,
            prn=prn,
            fs=fs,
            doppler_range_hz=doppler_range_hz,
            doppler_step_hz=doppler_step_hz,
            noncoherent_ms=noncoherent_ms,
            acquisition_threshold=acquisition_threshold,
            keep_surface=False,
        )
        for prn in prns
    ]
    results.sort(key=lambda r: r.peak_to_mean_snr, reverse=True)
    return results
