"""Synthetic 25 MHz multi-PRN GPS L1 C/A baseband I/Q generator for offline reproducibility.

Generates physically grounded complex baseband I/Q captures matching TEXBAT's
sampling rate (25 MSPS, interleaved int16-equivalent scale) across canonical scenarios:
- `CleanStatic`: multi-satellite authentic GPS L1 C/A + thermal AWGN.
- `CleanDynamic`: authentic GPS with specular multipath reflection + Doppler drift.
- `DS1_SwitchOver`: carrier/code switch-over with residual vestigial authentic signal.
- `DS2_OverpoweredTimePush`: +10 dB overpowered spoofing replica pulling code phase off.
- `DS3_MatchedPowerTimePush`: +1.3 dB matched-power spoofing replica at 0.45-chip lag
  (producing strong correlation-triangle widening & asymmetry with minimal C/N0 jump).
"""

from __future__ import annotations

import numpy as np

from .signal_processing import DEFAULT_FS_HZ, GPS_CA_CODE_LENGTH, GPS_CHIP_RATE_HZ, generate_ca_code

# Representative constellation visible in TEXBAT static captures (PRN 23 strongest)
DEFAULT_CONSTELLATION: tuple[tuple[int, float, float, float], ...] = (
    # (prn, doppler_hz, code_phase_chips, relative_amp)
    (23, -300.0, 104.2, 1.00),
    (3, 1450.0, 412.7, 0.86),
    (13, -1850.0, 738.1, 0.78),
    (16, 920.0, 289.5, 0.75),
    (19, -2420.0, 891.0, 0.71),
    (6, 610.0, 55.4, 0.68),
)


def _synthesize_prn_wave(
    prn: int,
    n_samples: int,
    fs: float,
    doppler_hz: float,
    code_phase_chips: float,
    carrier_phase_rad: float,
    doppler_rate_hz_per_s: float = 0.0,
) -> np.ndarray:
    """Synthesize complex baseband BPSK C/A waveform for one satellite."""
    chips = generate_ca_code(prn)
    t = np.arange(n_samples, dtype=np.float64) / fs
    # Instantaneous code phase in chips (including carrier-aiding Doppler scaling)
    code_rate = GPS_CHIP_RATE_HZ * (1.0 + doppler_hz / 1_575_420_000.0)
    chip_idx = np.floor(code_phase_chips + t * code_rate).astype(np.int64) % GPS_CA_CODE_LENGTH
    code_wave = chips[chip_idx]
    phase = carrier_phase_rad + 2.0 * np.pi * (doppler_hz * t + 0.5 * doppler_rate_hz_per_s * (t ** 2))
    return (code_wave * np.exp(1j * phase)).astype(np.complex64)


def generate_synthetic_iq_scenario(
    scenario: str,
    duration_ms: float = 4.0,
    fs: float = DEFAULT_FS_HZ,
    seed: int = 42,
) -> np.ndarray:
    """Generate a realistic multi-PRN complex64 GPS L1 C/A baseband capture.

    Amplitudes are scaled to match TEXBAT int16 ADC levels (~1500 RMS thermal noise),
    where GPS signals sit ~18-20 dB below the thermal noise floor prior to C/A
    despreading gain (10*log10(25000) ~= 44 dB).
    """
    rng = np.random.default_rng(seed)
    n_samples = int(round(fs * duration_ms * 1e-3))
    noise_sigma = 1450.0  # per I and Q branch -> matches TEXBAT int16 ~[-5000, +5000]

    # Baseband thermal noise (complex AWGN)
    iq = (
        rng.normal(0.0, noise_sigma, size=n_samples)
        + 1j * rng.normal(0.0, noise_sigma, size=n_samples)
    ).astype(np.complex64)

    # Authentic satellite nominal amplitude for CN0 ~ 49.5 dB-Hz:
    # SNR_coh(1ms) = A^2 * N_1ms / (2 * sigma^2) ~= 90 => A ~= sigma * sqrt(180 / N_1ms)
    n_1ms = fs * 1e-3
    base_amp = noise_sigma * np.sqrt(185.0 / n_1ms)

    key = scenario.lower().replace("_", "").replace("-", "")

    if key == "cleanstatic":
        for prn, dop, cphase, rel in DEFAULT_CONSTELLATION:
            phi0 = float(rng.uniform(0, 2 * np.pi))
            iq += (base_amp * rel) * _synthesize_prn_wave(prn, n_samples, fs, dop, cphase, phi0)

    elif key == "cleandynamic":
        dyn_constellation = (
            (18, 200.0, 949.8, 0.88),
            (22, -1100.0, 310.4, 0.82),
            (15, 1750.0, 620.1, 0.74),
            (9, -640.0, 145.6, 0.70),
            (3, 2150.0, 502.3, 0.64),
        )
        for prn, dop, cphase, rel in dyn_constellation:
            phi0 = float(rng.uniform(0, 2 * np.pi))
            direct = _synthesize_prn_wave(prn, n_samples, fs, dop, cphase, phi0, doppler_rate_hz_per_s=35.0)
            # Specular vehicle multipath reflection delayed by ~0.35 chips (-7 dB)
            multipath = 0.42 * _synthesize_prn_wave(
                prn, n_samples, fs, dop - 8.0, cphase + 0.35, phi0 + 1.1, doppler_rate_hz_per_s=35.0
            )
            iq += (0.86 * base_amp * rel) * (direct + multipath)

    elif key in ("ds1", "ds1switchover"):
        # Switch-over attack during capture: spoofer replaces authentic with slight code/phase offset
        # while a vestigial authentic component (-3.5 dB, shifted by 1.8 chips) lingers
        for prn, dop, cphase, rel in DEFAULT_CONSTELLATION:
            phi_auth = float(rng.uniform(0, 2 * np.pi))
            phi_spoof = phi_auth + 2.2
            auth = 0.52 * _synthesize_prn_wave(prn, n_samples, fs, dop, cphase, phi_auth)
            spoof = 1.08 * _synthesize_prn_wave(prn, n_samples, fs, dop - 100.0, cphase + 2.35, phi_spoof)
            iq += (base_amp * rel) * (auth + spoof)

    elif key in ("ds2", "ds2overpoweredtimepush"):
        # Overpowered (+10 dB => ~3.16x amplitude) time-push spoofer pulling code phase by +0.55 chips
        # plus elevated broadband transmitter noise floor (+1.2 dB)
        iq *= 1.15
        for prn, dop, cphase, rel in DEFAULT_CONSTELLATION:
            phi_auth = float(rng.uniform(0, 2 * np.pi))
            phi_spoof = phi_auth + 0.7
            auth = 1.00 * _synthesize_prn_wave(prn, n_samples, fs, dop, cphase, phi_auth)
            spoof = 2.85 * _synthesize_prn_wave(prn, n_samples, fs, dop - 100.0, cphase + 0.52, phi_spoof)
            iq += (base_amp * rel) * (auth + spoof)

    elif key in ("ds3", "ds3matchedpowertimepush"):
        # Matched-power (+1.3 dB => ~1.16x amplitude) time-push at +0.42 chips lag:
        # C/N0 barely changes, while correlation peak HPW widens and skews strongly!
        for prn, dop, cphase, rel in DEFAULT_CONSTELLATION:
            phi_auth = float(rng.uniform(0, 2 * np.pi))
            phi_spoof = phi_auth + 0.35
            auth = 1.00 * _synthesize_prn_wave(prn, n_samples, fs, dop, cphase, phi_auth)
            spoof = 1.16 * _synthesize_prn_wave(prn, n_samples, fs, dop, cphase + 0.42, phi_spoof)
            iq += (base_amp * rel) * (auth + spoof)

    else:
        raise ValueError(
            f"Unknown synthetic scenario {scenario!r}. Choose from "
            "'CleanStatic', 'CleanDynamic', 'DS1_SwitchOver', 'DS2_OverpoweredTimePush', 'DS3_MatchedPowerTimePush'."
        )

    return iq
