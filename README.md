# GPS Spoofing Detection on TEXBAT Recordings

Signal-processing and machine-learning pipeline for detecting spoofing attacks against civilian GPS signals, using raw RF recordings from the **TEXBAT** (Texas Spoofing Test Battery) dataset. The project extracts integrity features from the signal-acquisition stage and trains classifiers to label signal segments as *Clean* or *Spoofed*.

This work was developed for the Signals and Systems course (ES413) at CIn/UFPE.

## Problem

The open nature of civilian GPS makes its signals vulnerable to spoofing, in which a transmitter broadcasts counterfeit signals to deceive a receiver about its true position or time. Detecting such attacks is a classical engineering problem at the intersection of signal processing and security. This project builds a detector based on **signal-integrity monitoring and correlation-based analysis**, which are the natural primitives of the GPS acquisition chain.

## Dataset

- **TEXBAT** — Texas Spoofing Test Battery, University of Texas Radionavigation Laboratory.
- https://radionavlab.ae.utexas.edu/texbat/
- Raw RF recordings (`.bin`) with interleaved `int16` I/Q samples.
- Scenarios used in this work:
  - `CleanStatic.bin` — authentic static capture (baseline, no spoofing).
  - `ds1.bin` — switch-over attack: abrupt replacement of the authentic carriers by the spoofer's carriers, producing phase/code transients.
  - `ds2.bin` — overpowered time-push attack: the spoofer transmits code/phase-aligned replicas while shifting the receiver's time reference.

TEXBAT recordings are large and are not redistributed with this repository. Download the `.bin` files from the link above and place them in a `data/` directory alongside the notebook (the notebook exposes the expected paths near the top).

## Methodology

1. **Raw-signal ingestion** — memory-map the `.bin` file, extract a configurable number of samples, and convert the interleaved `int16` pairs into complex `I + jQ` baseband samples.
2. **Time- and frequency-domain inspection** — plot the time series and power spectrum of each capture to characterize noise, interference, and coarse structural differences between clean and spoofed recordings.
3. **GPS acquisition** — perform a 2D acquisition search over PRN, code phase, and Doppler shift using the C/A codes of candidate satellites, then evaluate the resulting correlation surface.
4. **Feature extraction** — for each acquired PRN, compute signal-quality features such as:
   - Correlation-peak shape metrics (height, width, asymmetry).
   - Carrier-to-noise ratio (C/N0) and its variation across channels.
5. **Classification** — Random Forest and Support Vector Machine classifiers are trained on the extracted features to label segments as *Clean* or *Spoofed*.
6. **Evaluation** — accuracy, precision, recall, F1-score, and per-class classification reports on a held-out test set.

## Summary of Results

| Classifier         | Test Accuracy |
|--------------------|---------------|
| Support Vector Machine | 100%     |
| Random Forest          | fails on this feature set |

The SVM achieved perfect precision, recall, and F1-score on both classes, while the Random Forest configuration used in the notebook did not generalize on this feature representation. These results are tightly coupled to the specific features extracted and to the limited set of TEXBAT scenarios considered; they should be interpreted as evidence that the correlation-based features carry strong signal, rather than as a general ranking of the two algorithms.

## Repository Layout

```
.
├── README.md
├── .gitignore
├── notebooks/
│   └── texbat_analysis.ipynb        # Full signal-processing and ML pipeline
└── docs/
    ├── project-proposal.pdf         # Original course assignment
    └── course-material.pdf          # Supporting course material
```

## How to Run

The notebook runs on Python 3.10+ with the scientific and ML stack:

```bash
pip install numpy scipy matplotlib scikit-learn jupyter
jupyter notebook notebooks/texbat_analysis.ipynb
```

Download the TEXBAT `.bin` files (`CleanStatic.bin`, `ds1.bin`, `ds2.bin`) from the TEXBAT website and update the data paths near the top of the notebook to point to the local copies.

## Author

- Artur Vinicius Pereira Fernandes

Centro de Informatica (CIn), Universidade Federal de Pernambuco (UFPE).

## References

- Humphreys, T. E. et al. *Assessing the Spoofing Threat: Development of a Portable GPS Civilian Spoofer.* ION GNSS 2008.
- Psiaki, M. L. and Humphreys, T. E. *GNSS Spoofing and Detection.* Proceedings of the IEEE, 104(6), 2016.
- Nature Scientific Reports article on GNSS spoofing detection: https://www.nature.com/articles/s41598-025-16853-1
- IEEE Xplore document 10566287: https://ieeexplore.ieee.org/document/10566287
- TEXBAT dataset: https://radionavlab.ae.utexas.edu/texbat/
