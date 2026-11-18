# Data Directory Layout

Large raw RF captures (`.bin`, ~40–45 GB per scenario) and intermediate GNSS-SDR correlator dumps (`.mat`, ~18 MB per scenario) are excluded from Git via `.gitignore`, while the compact extracted 1-second multi-channel SQM feature dataset is versioned under `data/processed/` so all machine-learning benchmarks and notebooks run immediately upon cloning.

```text
data/
├── processed/
│   ├── texbat_features_1s.csv       # 1-second multi-channel SQM & tracking features (CleanStatic, CleanDynamic, DS1-DS6)
│   └── evaluation_summary.json      # Metrics for Stratified, Temporal-Block, and Leave-One-Attack-Out evaluations
├── raw/                             # (Optional, gitignored) Place UT Austin TEXBAT .bin files here
│   ├── cleanStatic.bin
│   ├── cleanDynamic.bin
│   └── ds1.bin ... ds6.bin
└── texbat_gnsssdr/                  # (Optional, gitignored) Per-scenario GNSS-SDR HDF5 .mat tracking dumps
    ├── cleanStatic/tracking_ch_*.mat
    ├── cleanDynamic/tracking_ch_*.mat
    └── ds1/ ... ds6/
```

## Obtaining Raw TEXBAT Recordings

Download official `.bin` captures (25 MSPS interleaved `int16` I/Q) from the University of Texas Radionavigation Laboratory:
- https://radionavlab.ae.utexas.edu/texbat/

When `data/raw/*.bin` is not present locally, Stage 1 of the pipeline (`load_or_synthesize_raw_iq`) automatically synthesizes calibrated 25 MHz multi-PRN GPS L1 C/A baseband waveforms so acquisition, Welch PSD, and correlation-peak SQM analyses execute offline out-of-the-box.
