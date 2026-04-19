# PC-FMCW Laser Headlamp — ISCAI System

> **Research Assignment** · Principles of Telecommunication Systems  
> Democritus University of Thrace — Department of Electrical & Computer Engineering

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![License](https://img.shields.io/badge/License-Academic-lightgrey)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## Overview

This repository contains the simulation code and analysis for a research assignment based on the paper:

> S. Liu, T. Sun, X. Shu, J. Song and Y. Dong,  
> **"Phase-coded FMCW Laser Headlamp for Integrated Sensing, Communication, and Illumination"**  
> *IEEE Photonics Technology Letters*, 2025. DOI: [10.1109/LPT.2025.3649597](https://doi.org/10.1109/LPT.2025.3649597)

The paper proposes an **Integrated Sensing, Communication, and Illumination (ISCAI)** system for intelligent connected vehicles (ICV), built around a **Phase-Coded FMCW (PC-FMCW) laser headlamp** that simultaneously handles:

| Subsystem | Technology | Paper Performance |
|---|---|---|
| 📡 Communication | DPSK modulation on FMCW phase | 1 Gbps data rate |
| 🎯 Sensing (Radar) | FMCW Range-Doppler + 2D CA-CFAR | 3.8 cm ranging accuracy |
| 💡 Illumination | Adaptive Driving Beam (ADB) + Phosphor | SAE J3069 compliant |

The assignment is split into two parts:
- **Part A (6/10 pts):** Reproduce 6 key results from the paper
- **Part B (4/10 pts):** Propose and simulate 19 system improvement ideas

---

## Repository Structure

```
fmcw-laser-headlamp-enhancement/
│
├── scripts/                        ← Main Python simulation scripts
│   ├── iscai_reproduction.py       ← Part A: reproduces paper figures/results
│   ├── iscai_improvement.py        ← Part B: OFDM + Adaptive MHT (main proposals)
│   └── all_19_improvements.py      ← Part B: all 19 improvement ideas
│
├── results/                        ← Generated output figures
│   ├── iscai_results_reproduction.png   ← Part A: 9-subplot figure
│   ├── iscai_improvement_proposal.png   ← Part B: main proposal figure
│   └── all_19_improvements.png          ← Part B: 5×4 grid of all 19 ideas
│
├── README.md                       ← This file (English)
└── READMEgr.md                     ← Greek version
```

---

## Installation

**Requirements:** Python ≥ 3.8

```bash
pip install numpy scipy matplotlib scikit-learn
```

---

## Usage

```bash
# Part A — Reproduce paper results
python scripts/iscai_reproduction.py

# Part B — Main improvement proposals (OFDM + Adaptive MHT)
python scripts/iscai_improvement.py

# Part B — All 19 improvement ideas
python scripts/all_19_improvements.py
```

Each script generates and saves a figure to the `results/` folder.

---

## Part A — Result Reproduction

Six key results from the paper are reproduced in `iscai_reproduction.py`:

### 1. Range CRLB
Cramér-Rao Lower Bound for range estimation accuracy.

| Parameter | Value |
|---|---|
| SNR | 10 dB |
| Chirps M | 100 |
| Simulated σ_R | ~4.6 cm |
| Paper result | 3.8 cm ✓ |

> Small discrepancy is expected — the paper does not explicitly state M.

### 2. Range-Doppler Maps
Two scenarios reproducing paper Fig. 2:
- **(a) Well-separated targets:** 30 m / +5 m/s and 80 m / −8 m/s → distinct FFT peaks
- **(b) Closely spaced targets:** 50 m and 52.5 m at similar velocities → overlapping peaks

### 3. MHT-TBD Tracking
Reproduces the Multidimensional Hough Transform Track-Before-Detect algorithm (paper Fig. 4):
- Scenario 1: 2 linear tracks in Gaussian noise + clutter — parameter errors: 0.1251, 0.2348 units
- Scenario 2: 1 linear + 1 non-linear track in dense clutter — mean deviation: **1.6787 units** ✓

### 4. Adaptive Driving Beam (ADB)
- Oncoming vehicle: host @ 40 km/h, oncoming @ 30 km/h, initial separation 150 m
- Multiple vehicles: host @ 50 km/h, 2 preceding vehicles with 30 m initial spacing

### 5. DPSK BER
`BER = 0.5 · exp(−SNR)` — at SNR = 10 dB: BER ≈ 2.27 × 10⁻⁵

### 6. Group Delay Filter (GDF)
Restores the LFM structure after DPSK phase coding: `H_g(ω) = exp(−j · ω · τ_g(ω))`

---

## Part B — 19 Improvement Ideas

Improvements are organized in 5 categories:

### Category 1 — Communication

| # | Idea | Key Improvement |
|---|---|---|
| 1 | **OFDM-FMCW** | 2–4× throughput, better BER |
| 2 | **LDPC / Turbo Coding** | +5.5 dB coding gain |
| 3 | **Polarization Multiplexing** | 2× throughput (2 Gbps), no extra bandwidth |
| 4 | **Adaptive MMSE Equalization** | Robust under atmospheric turbulence |

### Category 2 — Sensing / Radar

| # | Idea | Key Improvement |
|---|---|---|
| 5 | **MUSIC Superresolution** | 10× resolution beyond FFT limit |
| 6 | **Compressed Sensing FMCW** | 70% fewer chirps, same resolution |
| 7 | **Wideband FMCW (B = 50 GHz)** | 3.8 cm → 1.85 cm ranging error |
| 8 | **CNN Range-Doppler (NN-CFAR)** | Better ROC, robust detection |

### Category 3 — Tracking / Detection

| # | Idea | Key Improvement |
|---|---|---|
| 9 | **Particle Filter TBD** | 1.60 units vs MHT 3.24 units |
| 10 | **JPDA** | 80.2% accuracy vs 64.5% (AND-logic) |
| 11 | **Adaptive MHT (Otsu Threshold)** | FAR −80%, TDR = 100% maintained |
| 12 | **LSTM Track Prediction** | Better on non-linear maneuvers |

### Category 4 — ADB Illumination

| # | Idea | Key Improvement |
|---|---|---|
| 13 | **Micro-LED Array** | 0.78° vs 3.0° angular error |
| 14 | **Semantic ADB** | Per-class safety margins (pedestrian/cyclist/vehicle) |
| 15 | **LiDAR-guided ADB** | Eliminates camera offset error (Δy = 0) |

### Category 5 — System / Waveform Design

| # | Idea | Key Improvement |
|---|---|---|
| 16 | **Cognitive ISAC** | Dynamic sensing/comms resource allocation |
| 17 | **Frequency Hopping FMCW** | Interference: 90% → 7.6% (5 vehicles) |
| 18 | **Waveform Optimization (Pareto)** | Pareto-optimal sensing/comms trade-off |
| 19 | **Optical MIMO Beamforming** | 5 → 32 bits/s/Hz (8×8 MIMO) |

### Results Summary

| Metric | Paper Baseline | Best Proposal | Gain |
|---|---|---|---|
| Data rate | 1 Gbps | 4 Gbps (OFDM-16QAM) | **4×** |
| Ranging error | 3.8 cm | 1.85 cm (Wideband) | **2×** |
| False alarm rate | baseline | −80% (Adaptive MHT) | **5×** |
| Association accuracy | 64.5% | 80.2% (JPDA) | **+24%** |
| ADB precision | 3.0° | 0.78° (Micro-LED) | **4×** |
| Interference probability | ~90% | 7.6% (Freq. Hopping) | **12×** |
| Channel capacity | 5 b/s/Hz | 32 b/s/Hz (8×8 MIMO) | **6×** |

---

## Key System Parameters

```
fc  = 193.4 THz     laser carrier frequency (λ ≈ 1551 nm)
B   = 10 GHz        chirp bandwidth
T   = 10 μs         chirp period
Rb  = 1 Gbps        data rate
μ   = B/T = 10¹⁵    chirp rate (Hz/s)
```

---

## References

1. S. Liu, T. Sun, X. Shu, J. Song, Y. Dong — *"Phase-coded FMCW Laser Headlamp for ISCAI"*, IEEE PTL, 2025. DOI: 10.1109/LPT.2025.3649597
2. F. Liu et al. — *"ISAC: Towards dual functional wireless networks for 6G"*, IEEE JSAC, 2022.
3. U. Kumbul et al. — *"Phase-coded FMCW for coherent MIMO radar"*, IEEE Trans. MTT, 2023.
4. U. Kumbul et al. — *"Smoothed phase-coded FMCW: waveform properties"*, IEEE Trans. AES, 2023.
5. Y. Zhou et al. — *"A 3D Hough Transform-based TBD technique"*, Sensors, 2019.
6. W. Li et al. — *"Greedy integration based multi-frame detection"*, IEEE Trans. VT, 2023.
7. SAE International — *"Adaptive Driving Beam (ADB) system requirements"*, SAE J3069, 2021.
8. Q. Zheng et al. — *"A target detection scheme for range-Doppler FMCW radar"*, IEEE TIM, 2021.

---

*Democritus University of Thrace · Department of Electrical & Computer Engineering*  
*Course: Principles of Telecommunication Systems*
