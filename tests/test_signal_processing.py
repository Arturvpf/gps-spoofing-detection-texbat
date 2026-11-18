"""Unit tests for GPS L1 C/A Gold code generation, rectangular sampling, .bin I/Q loading, and 2D PCPS acquisition."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from texbat_spoofing.signal_processing import (
    GPS_CA_CODE_LENGTH,
    acquire_prn_pcps,
    circular_exclusion_mask,
    compute_psd_and_flatness,
    generate_ca_code,
    load_iq_from_bin,
    sample_ca_code,
)
from texbat_spoofing.synthetic_texbat import generate_synthetic_iq_scenario


def test_ca_code_length_balance_and_orthogonality() -> None:
    """IS-GPS-200 Gold codes have length 1023, values in {-1, +1}, and bounded cross-correlation."""
    c1 = generate_ca_code(1)
    c23 = generate_ca_code(23)

    assert c1.shape == (GPS_CA_CODE_LENGTH,)
    assert set(np.unique(c1)).issubset({-1.0, 1.0})

    # Verify IS-GPS-200 Table 3-Ia octal first-10-chip phase for PRN 1 (octal 1440 = 1100100000)
    # BPSK mapping: bit 0 -> +1, bit 1 -> -1 => [-1, -1, +1, +1, -1, +1, +1, +1, +1, +1]
    expected_prn1_first10 = np.array([-1, -1, 1, 1, -1, 1, 1, 1, 1, 1], dtype=np.float32)
    np.testing.assert_array_equal(c1[:10], expected_prn1_first10)

    # Gold code unnormalized circular cross-correlation belongs to {-65, -1, +63} (magnitude <= 65)
    cross_corr = np.abs(np.fft.ifft(np.fft.fft(c1) * np.conj(np.fft.fft(c23))))
    assert np.max(cross_corr) <= 65.5


def test_invalid_prn_raises() -> None:
    with pytest.raises(ValueError, match="Invalid GPS PRN"):
        generate_ca_code(0)
    with pytest.raises(ValueError, match="Invalid GPS PRN"):
        generate_ca_code(33)


def test_sample_ca_code_repeats_periodically_without_gibbs_ringing() -> None:
    """Regression test: multi-ms sampling must repeat the 1 ms C/A sequence, not stretch it."""
    fs = 4_092_000.0  # exactly 4 samples per chip -> 4092 samples per ms
    one_ms = sample_ca_code(23, fs=fs, duration_ms=1.0)
    ten_ms = sample_ca_code(23, fs=fs, duration_ms=10.0)

    assert one_ms.shape == (4092,)
    assert ten_ms.shape == (40920,)
    # Rectangular BPSK values must remain strictly +1 or -1 (no FFT sinc overshoot)
    assert set(np.unique(ten_ms)) == {-1.0, 1.0}
    # Every 1 ms slice must equal the 1 ms reference period
    for k in range(10):
        np.testing.assert_array_equal(ten_ms[k * 4092 : (k + 1) * 4092], one_ms)


def test_load_iq_from_bin_with_start_time_and_dc_removal(tmp_path: Path) -> None:
    fs = 10_000.0
    # Create 2 seconds (20,000 complex samples = 40,000 int16 values)
    t = np.arange(20_000, dtype=np.float64) / fs
    i_vals = np.round(500.0 * np.cos(2 * np.pi * 250.0 * t) + 25.0).astype(np.int16)
    q_vals = np.round(500.0 * np.sin(2 * np.pi * 250.0 * t) - 15.0).astype(np.int16)
    interleaved = np.empty(40_000, dtype=np.int16)
    interleaved[0::2] = i_vals
    interleaved[1::2] = q_vals

    bin_file = tmp_path / "test_capture.bin"
    interleaved.tofile(bin_file)

    iq = load_iq_from_bin(bin_file, num_samples=5_000, start_time_s=1.0, fs=fs, remove_dc=True)
    assert iq.shape == (5_000,)
    assert iq.dtype == np.complex64
    assert abs(np.mean(iq)) < 1.0


def test_pcps_acquisition_recovers_prn_doppler_and_code_phase() -> None:
    fs = 5_000_000.0  # 5 MSPS fast test
    iq = generate_synthetic_iq_scenario("CleanStatic", duration_ms=2.0, fs=fs, seed=7)

    acq_present = acquire_prn_pcps(
        iq, prn=23, fs=fs, doppler_range_hz=3000.0, doppler_step_hz=100.0, noncoherent_ms=2
    )
    acq_absent = acquire_prn_pcps(
        iq, prn=31, fs=fs, doppler_range_hz=3000.0, doppler_step_hz=100.0, noncoherent_ms=2
    )

    assert acq_present.acquired is True
    assert abs(acq_present.doppler_hz - (-300.0)) <= 100.0
    # Code phase synthesized at 104.2 chips => receiver lag is (1023 - 104.2) mod 1023 = 918.8 chips
    expected_lag_chips = (1023.0 - 104.2) % 1023.0
    assert abs(acq_present.code_phase_chips - expected_lag_chips) < 0.6
    assert 45.0 <= acq_present.cn0_dbhz <= 55.0
    assert acq_absent.acquired is False


def test_circular_exclusion_mask_wraps_around_boundaries() -> None:
    mask = circular_exclusion_mask(n_samples=100, center_idx=2, exclusion_radius=5)
    assert mask[2] is np.bool_(False)
    assert mask[7] is np.bool_(False)
    assert mask[97] is np.bool_(False)
    assert mask[8] is np.bool_(True)
    assert mask[96] is np.bool_(True)


def test_psd_and_spectral_flatness_white_noise_vs_tone() -> None:
    rng = np.random.default_rng(123)
    white = (rng.normal(size=32768) + 1j * rng.normal(size=32768)).astype(np.complex64)
    t = np.arange(32768) / 1e6
    tone = (10.0 * np.exp(1j * 2 * np.pi * 1e5 * t) + 0.1 * white).astype(np.complex64)

    m_white = compute_psd_and_flatness(white, fs=1e6)
    m_tone = compute_psd_and_flatness(tone, fs=1e6)

    assert m_white.spectral_flatness > 0.85
    assert m_tone.spectral_flatness < 0.25
