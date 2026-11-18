# GPS Spoofing Detection on TEXBAT Recordings

End-to-end **signal-processing** and **machine-learning** pipeline for detecting civilian GPS L1 C/A spoofing attacks on the **TEXBAT** (*Texas Spoofing Test Battery*) dataset. Developed for the *Signals and Systems* course (**ES413**) at **CIn / UFPE**.

---

## Overview

Civilian GPS L1 C/A signals ($1575.42\text{ MHz}$) are unencrypted and arrive at the receiver ~19 dB below the thermal noise floor prior to PRN code despreading, making standard receivers vulnerable to counterfeit signal injection (*spoofing*).

This project implements a **two-stage detection architecture** grounded in GPS correlation physics and Signal Quality Monitoring (**SQM**):

1. **Stage 1 — 25 MSPS Baseband RF Acquisition & Correlation-Peak SQM (`signal_processing.py`, `sqm_features.py`):**
   - Memory-mapped ingestion of interleaved `int16` I/Q `.bin` recordings (with time-offset support targeting active post-onset spoofing windows, plus an automatic physically calibrated 25 MHz multi-PRN GPS L1 C/A synthesizer when ~45 GB raw `.bin` files are absent locally).
   - Two-sided Welch Power Spectral Density (PSD), Wiener spectral flatness, and wideband power monitoring.
   - Exact IS-GPS-200 Gold C/A code generation (PRNs 1–32) with zero-order-hold rectangular sampling and 2D FFT **Parallel Code-Phase Search (PCPS)** over Doppler $\times$ Code Phase.
   - Waveform SQM metrics: **Ratio of the Correlation Peak (RCP)**, sub-sample interpolated **Half-Peak Width (HPW, chips)**, **Leading/Trailing Asymmetry (Skew)**, **Early–Prompt–Late Ratio & Delta SQM**, and Rayleigh-calibrated **$C/N_0$ ($\text{dB-Hz}$)**.

2. **Stage 2 — Continuous Multi-Channel TEXBAT Benchmark (`data_loader.py`, `classifiers.py`):**
   - Extracts **15 multi-channel SQM and carrier/code tracking features** over **3,303 1-second receiver epochs** across all **8 core TEXBAT scenarios** (`cleanStatic`, `cleanDynamic`, `ds1`–`ds6`).
   - Trains and evaluates leak-free `StandardScaler` pipelines for **Random Forest**, **Support Vector Machine (SVM)**, and **MLP Neural Network** under three validation regimes:
     1. *Stratified Hold-Out (70/30) + 5-Fold Stratified CV*;
     2. *Chronological Temporal Block Split* with a 10-second guard gap (eliminating adjacent-window autocorrelation leakage);
     3. *Leave-One-Attack-Out (LOAO)* across `ds1`–`ds6` (measuring zero-shot generalization to unseen spoofing attack types).

---

## TEXBAT Scenarios Covered

| Scenario | Platform | Attack Type | Power Advantage | Attack Onset ($t_{\text{onset}}$) |
|---|---|---|---:|---:|
| `cleanStatic` | Static | Authentic baseline (no spoofing) | — | — |
| `cleanDynamic` | Dynamic | Authentic moving-vehicle baseline (with multipath) | — | — |
| `ds1` | Static | Abrupt carrier/code **Switch-Over** | $0.0\text{ dB}$ | $125\text{ s}$ |
| `ds2` | Static | **Overpowered Time-Push** | $+10.0\text{ dB}$ | $110\text{ s}$ |
| `ds3` | Static | **Matched-Power Time-Push** | $+1.3\text{ dB}$ | $120\text{ s}$ |
| `ds4` | Static | **Matched-Power Position-Push** | $+0.4\text{ dB}$ | $115\text{ s}$ |
| `ds5` | Dynamic | **Overpowered Time-Push** | $+9.9\text{ dB}$ | $105\text{ s}$ |
| `ds6` | Dynamic | **Matched-Power Position-Push** | $+0.8\text{ dB}$ | $105\text{ s}$ |

---

## Summary of Results

Evaluated on **3,303 1-second multi-channel epochs** (`1,356 Clean` / `1,947 Spoofed`, excluding an $\pm 8\text{ s}$ onset transition guard band):

### 1. Standard & Leakage-Free Temporal Block Evaluations

| Protocol | Classifier | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| **Stratified 70/30 + 5-Fold CV** | **Random Forest** | **99.90%** | **99.83%** | **100.00%** | **0.9991** | **1.0000** |
| Stratified 70/30 + 5-Fold CV | **SVM (Linear)** | 99.80% | 99.66% | 100.00% | 0.9983 | 0.9997 |
| Stratified 70/30 + 5-Fold CV | **MLP (64, 32)** | 99.39% | 99.49% | 99.49% | 0.9949 | 0.9998 |
| **Temporal Block 70/30 (10s gap)** | **Random Forest** | **100.00%** | **100.00%** | **100.00%** | **1.0000** | **1.0000** |
| Temporal Block 70/30 (10s gap) | **SVM (Linear)** | 99.79% | 100.00% | 99.65% | 0.9982 | 1.0000 |
| Temporal Block 70/30 (10s gap) | **MLP (32, 16)** | 98.84% | 99.82% | 98.24% | 0.9902 | 0.9994 |

### 2. Zero-Shot Leave-One-Attack-Out (LOAO) Generalization (`DS1`–`DS6`)

| Classifier | Mean Unseen-Attack Recall | Clean False-Alarm Rate | Mean Fold F1 | Mean Fold ROC-AUC |
|---|---:|---:|---:|---:|
| **Random Forest** | **96.64%** | **0.00%** | **0.9817** | **1.0000** |
| **SVM (RBF)** | 96.51% | **0.00%** | 0.9795 | 0.9995 |
| **MLP (Neural Net)** | 95.57% | **0.00%** | 0.9750 | 0.9988 |

> **Key Physical Finding:** Unseen-attack recall reaches **97%–100%** on `ds1`, `ds2`, `ds3`, `ds5`, and `ds6`, while dipping to **83.6%–86.9%** (`ROC-AUC = 0.9998`) on `ds4` (*Static Matched-Power Position-Push*, $+0.4\text{ dB}$). During the initial lock-alignment window ($115\text{--}170\text{ s}$) of `ds4`, the counterfeit signal has near-zero code-phase offset and minimal power excess; once position pull-off exceeds $\approx 0.15\text{ chips}$ ($t > 172\text{ s}$), `sqm_ratio` surges and detection reaches **100%**.

---

## Repository Layout

```text
.
├── README.md
├── pyproject.toml                       # Editable package configuration & pytest setup
├── requirements.txt                     # Scientific Python & ML dependencies
├── data/
│   ├── README.md                        # Dataset documentation & directory instructions
│   └── processed/
│       ├── texbat_features_1s.csv       # Pre-extracted 1s multi-channel TEXBAT features (3,303 rows)
│       └── evaluation_summary.json      # Reproducible evaluation metrics across all 3 protocols
├── docs/
│   ├── project-proposal.pdf             # Original ES413 course project specification
│   ├── presentation.pdf                 # Final presentation slides
│   └── relatorio-final.md               # Comprehensive technical report (in Portuguese)
├── notebooks/
│   └── texbat_analysis.ipynb            # Executed end-to-end signal-processing, SQM & ML notebook
├── scripts/
│   └── run_pipeline.py                  # CLI entry point for feature extraction & ML evaluation
├── src/
│   └── texbat_spoofing/
│       ├── __init__.py
│       ├── signal_processing.py         # C/A Gold codes, .bin I/Q loader, Welch PSD, 2D FFT PCPS
│       ├── sqm_features.py              # RCP, sub-sample HPW, Skew, E-P-L Ratio/Delta SQM
│       ├── synthetic_texbat.py          # Calibrated 25 MHz multi-PRN GPS + spoofing I/Q generator
│       ├── data_loader.py               # TEXBAT scenario metadata, .mat extractor & CSV loader
│       ├── classifiers.py               # Leak-free SVM/RF/MLP pipelines & Stratified/Temporal/LOAO eval
│       └── cli.py                       # Command-line runner
└── tests/
    ├── test_signal_processing.py        # IS-GPS-200 C/A code, ZOH sampling & PCPS acquisition tests
    ├── test_sqm_features.py             # Correlation-peak geometry, RCP & spoofing distortion tests
    └── test_classifiers.py              # Guard-gap temporal split & ML pipeline tests
```

---

## Quick Start

### 1. Install Dependencies

```bash
python -m venv .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

### 2. Run the Full CLI Benchmark & 25 MHz I/Q Demo

Runs immediately out-of-the-box using the versioned `data/processed/texbat_features_1s.csv` dataset and calibrated 25 MHz baseband synthesizer (or local `.bin` / `.mat` files if placed under `data/`):

```bash
python scripts/run_pipeline.py --demo-iq
```

### 3. Explore the Interactive Notebook

```bash
jupyter notebook notebooks/texbat_analysis.ipynb
```

### 4. Run Automated Tests

```bash
pytest -v
```

---

## Authors

- **Artur Vinicius Pereira Fernandes**
- **Isabela Maria de Moura Nascimento**
- **Karina Lima de Oliveira**

Centro de Informática (CIn), Universidade Federal de Pernambuco (UFPE).

## References

- Humphreys, T. E. et al. *Assessing the Spoofing Threat: Development of a Portable GPS Civilian Spoofer.* ION GNSS 2008.
- Humphreys, T. E. et al. *The Texas Spoofing Test Battery: Toward a Standard for Evaluating GPS Signal Authentication Techniques.* ION GNSS 2012.
- Psiaki, M. L. & Humphreys, T. E. *GNSS Spoofing and Detection.* Proceedings of the IEEE, 104(6), 1258–1270, 2016.
- Shafique, A., Mehmood, A., & Elhadef, M. *Detecting Signal Spoofing Attack in UAVs Using Machine Learning Models.* IEEE Access, 9, 2021.
- TEXBAT dataset: https://radionavlab.ae.utexas.edu/texbat/
