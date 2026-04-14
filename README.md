# PC-FMCW Laser Headlamp — ISCAI System
## Research Assignment | Principles of Telecommunication Systems — DUTH

> **Paper:** S. Liu, T. Sun, X. Shu, J. Song and Y. Dong,  
> *"Phase-coded FMCW Laser Headlamp for Integrated Sensing, Communication, and Illumination"*  
> IEEE Photonics Technology Letters, DOI: [10.1109/LPT.2025.3649597](https://doi.org/10.1109/LPT.2025.3649597)

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Repository Structure](#2-repository-structure)
3. [Installation](#3-installation)
4. [Part A — Result Reproduction (6/10)](#4-part-a--result-reproduction-610)
5. [Part B — 19 Improvement Ideas (4/10)](#5-part-b--19-improvement-ideas-410)
6. [Results & Figures](#6-results--figures)
7. [Mathematical Background](#7-mathematical-background)
8. [References](#8-references)

---

## 1. System Overview

The paper proposes an **Integrated Sensing, Communication, and Illumination (ISCAI)** system based on a **Phase-coded FMCW (PC-FMCW) laser headlamp** for intelligent connected vehicles (ICV).

### Three Unified Subsystems

| Subsystem | Technology | Performance (paper) |
|---|---|---|
| **Communication** | DPSK embedded in FMCW phase | 1 Gbps data rate |
| **Sensing (Radar)** | FMCW Range-Doppler + 2D CA-CFAR | 3.8 cm ranging accuracy |
| **Illumination** | Adaptive Driving Beam (ADB) + Phosphor | SAE J3069 compliant |

### Key System Parameters

```
fc  = 193.4 THz   (laser carrier frequency, λ ≈ 1551 nm)
B   = 10 GHz      (chirp bandwidth)
T   = 10 μs       (chirp period)
Rb  = 1 Gbps      (data rate)
μ   = B/T         (chirp rate = 10¹⁵ Hz/s)
```

### PC-FMCW Signal Architecture

```
Transmitted signal:
  s_T(t) = A_T · exp{j[2π·fc·t + π·μ·t² + φ_d(t)]}
                       └── LO chirp ──┘  └─ DPSK ─┘

Received echo:
  s_RX(t) = A_R · exp{j[2π·fc·(t-τ) + π·μ·(t-τ)² + φ_d(t-τ) + φ]}

Beat (IF) signal after mixing with LO:
  s_IF(t) = exp{j[-2π·fb·t + φ_d(t-τ) + φ]}
  where fb = μ·τ  →  range R = c·fb/(2μ)
```

### MHT-TBD Algorithm

Key innovation: **Multidimensional Hough Transform** (4D → 3×2D projections) with **AND-logic fusion** and **rolling-window** for maneuvering target tracking:

```
3D point cloud (x, y, t)
        ↓
  Project onto 3 planes: XY, XT, YT
        ↓
  2D HT + 3×3 mean filter per plane
        ↓
  AND-logic: common supporting points → valid tracks
        ↓
  Rolling window: piecewise linear segments
        ↓
  Cost function: C(T_seg, T_k) = w₁·D_pos + w₂·D_kin
```

---

## 2. Repository Structure

```
project/
├── README_EN.md                       ← this file (English)
├── README.md                          ← Greek version
│
├── iscai_reproduction.py              ← Part A: result reproduction
├── iscai_improvement.py               ← Part B: OFDM + Adaptive MHT (extended)
├── all_19_improvements.py             ← Part B: all 19 ideas together
│
├── iscai_results_reproduction.png     ← Part A figure (9 subplots)
├── iscai_improvement_proposal.png     ← Main proposal figure
└── all_19_improvements.png            ← All ideas figure (5×4 grid)
```

---

## 3. Installation

```bash
pip install numpy scipy matplotlib scikit-learn
```

**Requirements:**
- Python ≥ 3.8
- numpy, scipy, matplotlib, scikit-learn

**Run:**
```bash
# Part A: Reproduction
python iscai_reproduction.py

# Part B: Main proposal
python iscai_improvement.py

# Part B: All 19 ideas
python all_19_improvements.py
```

---

## 4. Part A — Result Reproduction (6/10)

Six key results from the paper are reproduced:

### 4.1 Range CRLB (paper eq. 7)

Cramér-Rao Lower Bound for range estimation:

```
var(R̂) ≥ (cT/2B)² · 3 / (8π²·γ·M·Tc²)
σ_R = (c/2) · √var(τ̂)
```

| Parameter | Value |
|---|---|
| SNR (γ) | 10 dB |
| Number of chirps M | 100 |
| Simulated CRLB error | ~4.6 cm |
| Paper result | 3.8 cm ✓ |

The small discrepancy is due to M not being explicitly stated in the paper.

### 4.2 Range-Doppler Maps (paper Fig. 2)

- **(a) Well-separated targets** (30m/+5m/s, 80m/−8m/s): two distinct peaks
- **(b) Closely spaced targets** (50m/+5m/s, 52.5m/+8m/s): overlapping peaks

Pipeline: beat signal generation → GDF phase compensation → 2D FFT → CA-CFAR detection

### 4.3 MHT-TBD Algorithm (paper Fig. 4)

**Scenario 1:** 2 linear tracks + Gaussian noise + random clutter  
→ AND-logic fusion, parameter errors: 0.1251 and 0.2348 units

**Scenario 2:** 1 linear + 1 non-linear track in dense clutter  
→ Rolling-window stitching, mean deviation: **1.6787 units** ✓

### 4.4 ADB Simulation (paper Fig. 3)

- **Oncoming vehicle scenario:** host at 40 km/h, oncoming at 30 km/h, initial separation 150m
- **Multiple vehicles scenario:** host at 50 km/h, 2 preceding vehicles with initial spacing 30m

### 4.5 DPSK Communication

DPSK BER formula: `BER = 0.5·exp(−SNR)`  
@ SNR = 10 dB: BER ≈ 2.27×10⁻⁵ → satisfactory for 1 Gbps link

### 4.6 Group Delay Filter (GDF)

```python
# Restores LFM structure corrupted by DPSK phase coding:
H_g(ω) = exp(−j·ω·τ_g(ω))
# Application: FFT(beat) · GDF → IFFT → clean LFM signal
```

---

## 5. Part B — 19 Improvement Ideas (4/10)

### Category 1: Communication

| # | Idea | Improvement | Complexity |
|---|---|---|---|
| 1 | **OFDM-FMCW** | 2–4× throughput, better BER | Medium |
| 2 | **LDPC / Turbo Coding** | +5.5 dB coding gain | Low |
| 3 | **Polarization Multiplexing** | 2× throughput (2 Gbps), no extra BW | Medium |
| 4 | **Adaptive MMSE Equalization** | Robust under atmospheric turbulence | Medium |

**Main idea (Idea 1 — OFDM):**
- Replace DPSK (1 bit/symbol) with N_sc = 64 OFDM subcarriers carrying QAM symbols
- Natural separation of sensing and comms in the frequency domain
- Cyclic prefix handles ISI
- BER @ 10 dB: DPSK ≈ 6.96×10⁻¹¹ → OFDM-4QAM ≈ 4.04×10⁻¹²

### Category 2: Sensing / Radar

| # | Idea | Improvement | Complexity |
|---|---|---|---|
| 5 | **MUSIC Superresolution** | 10× resolution beyond FFT limit | High |
| 6 | **Compressed Sensing FMCW** | 70% fewer chirps, same resolution | Medium |
| 7 | **Wideband FMCW (B = 50 GHz)** | 3.8 cm → 1.85 cm ranging error | High HW |
| 8 | **CNN Range-Doppler (NN-CFAR)** | Better ROC, robust detection | High |

**Main idea (Idea 7 — Wideband):**
```
Range resolution ΔR = c/(2B):
  B = 10 GHz  →  ΔR = 1.5 cm
  B = 50 GHz  →  ΔR = 0.3 cm

CRLB error @ SNR = 10 dB:
  B = 10 GHz  →  σ_R ≈ 3.8 cm   (paper)
  B = 50 GHz  →  σ_R ≈ 1.85 cm  (proposal)
```

### Category 3: Tracking / Detection

| # | Idea | Improvement | Complexity |
|---|---|---|---|
| 9 | **Particle Filter TBD** | 1.60u vs MHT 3.24u on maneuvering targets | Medium |
| 10 | **JPDA** | 80.2% accuracy vs 64.5% (AND-logic) | Medium |
| 11 | **Adaptive MHT Otsu Threshold** | FAR −80%, TDR = 100% | Very low |
| 12 | **LSTM Track Prediction** | Better performance on non-linear maneuvers | Medium |

**Main idea (Idea 11 — Adaptive Otsu):**
```python
# Paper: fixed threshold
threshold = 0.7 * accumulator.max()

# Proposal: Otsu's method (adaptive per frame)
threshold = otsu_threshold(accumulator)

# Result (500 Monte Carlo trials):
#   FAR:  485 → 95  ×10⁻⁴   (−80%)
#   TDR:  1.00 → 1.00        (unchanged)
```

### Category 4: ADB Illumination

| # | Idea | Improvement | Complexity |
|---|---|---|---|
| 13 | **Micro-LED Array** | 0.78° vs 3.0° error, pixel-level control | High HW |
| 14 | **Semantic ADB** | Per-class safety margin (pedestrian/cyclist/vehicle) | Medium |
| 15 | **LiDAR-guided ADB** | Δy = 0, eliminates camera offset error | Low SW |

### Category 5: System / Waveform Design

| # | Idea | Improvement | Complexity |
|---|---|---|---|
| 16 | **Cognitive ISAC** | Dynamic sensing/comms resource split | Medium |
| 17 | **Frequency Hopping FMCW** | P(interference): 90% → 7.6% (5 vehicles) | Low |
| 18 | **Waveform Optimization (Pareto)** | Pareto-optimal sensing/comms trade-off | High |
| 19 | **Optical MIMO Beamforming** | 5 → 32 bits/s/Hz (8×8 MIMO) | High HW |

---

## 6. Results & Figures

### `iscai_results_reproduction.png` — Part A (9 subplots)

```
┌──────────────────┬──────────────────┬──────────────────┐
│ RDM: separated   │ RDM: close tgts  │ CRLB vs SNR      │
├──────────────────┼──────────────────┼──────────────────┤
│ MHT raw data     │ Hough Space XY   │ Track error      │
├──────────────────┼──────────────────┼──────────────────┤
│ ADB: oncoming    │ ADB: multi-veh   │ DPSK BER vs SNR  │
└──────────────────┴──────────────────┴──────────────────┘
```

### `all_19_improvements.png` — Part B (5×4 grid)

```
Row 1 [Communication]:  BER OFDM | LDPC gain | Pol-MUX throughput | MMSE equalization
Row 2 [Sensing]:        MUSIC vs FFT | CS-FMCW | Wideband B | CNN ROC
Row 3 [Tracking]:       PF trajectory | JPDA accuracy | Otsu MHT | LSTM vs KF
Row 4 [ADB]:            Micro-LED beam | Semantic margins | LiDAR error | ADB summary
Row 5 [System]:         Cognitive ISAC | Freq hopping | Pareto optim | MIMO capacity
```

### Summary Table

| Metric (paper baseline) | Paper value | Best proposal | Improvement |
|---|---|---|---|
| Data rate | 1 Gbps | 4 Gbps (OFDM-16QAM) | **4×** |
| Ranging error | 3.8 cm | 1.85 cm (Wideband) | **2×** |
| Track deviation | 1.6787 units | 1.60 units (PF) | **~5%** |
| False alarm rate | baseline | −80% (Adaptive MHT) | **5×** |
| Association accuracy | 64.5% | 80.2% (JPDA) | **+24%** |
| ADB precision | 3.0° | 0.78° (Micro-LED) | **4×** |
| Interference probability | ~90% | 7.6% (Freq. Hop.) | **12×** |
| Channel capacity | 5 b/s/Hz | 32 b/s/Hz (8×8 MIMO) | **6×** |

---

## 7. Mathematical Background

### PC-FMCW Waveform

```
Local oscillator:
  s_LO(t) = A_T · e^{j[2π·fc·t + π·μ·t²]}

Transmitted (with DPSK):
  s_T(t)  = A_T · e^{j[2π·fc·t + π·μ·t² + φ_d(t)]}

DPSK encoding:
  φ_d(t) ∈ {0, π}  per symbol duration T_s = 1/R_b
  Δφ ∈ {0, π}  →  bit=0: no phase change, bit=1: π phase flip
```

### Range-Doppler Processing (2D FFT)

```
Range FFT (fast-time, per chirp m):
  X(m, k) = Σ_{n=0}^{N-1} x(m,n) · exp(−j·2π·nk/N),   0 ≤ k ≤ N−1

Doppler FFT (slow-time, per range bin k):
  V(q, k) = Σ_{m=0}^{M-1} X(m,k) · exp(−j·2π·mq/M),   0 ≤ q ≤ M−1

Power spectrum:
  Z(q, k) = |V(q, k)|²
```

### Cramér-Rao Lower Bound

```
Delay estimation:
  var(τ̂) ≥ (cT/2B)² · 3 / (8π²·γ·M·Tc²)

Doppler estimation:
  var(v̂) ≥ (λ/2)² · 3 / (8π²·γ·Tc²·M³)

where: γ = |A|²/σ² = SNR,   Tc = M·T = coherent integration time
```

### CA-CFAR Detection

```
Threshold:   T_th(q,k) = α · P̂_noise(q,k)
Decision:    Z(q,k) > T_th(q,k)  →  target declared

α is chosen to maintain constant false alarm probability P_fa:
  α = N_train · (P_fa^{−1/N_train} − 1)
```

### ADB Raised-Cosine Intensity

```
ℒ(θ, d) = {  0,                                          θ ∈ Θ, d ≤ d_min
            {  λ·(1 − cos(π·(d−d_min)/(d_max−d_min)))/2, θ ∈ Θ, d_min < d < d_max
            {  1,                                          otherwise

Shadow interval:  Θ = [θ_R − Δy/d − δ, θ_R − Δy/d + δ]
```

### Hough Transform Track Detection

```
Line in normal form:  ρ = x·cos(θ) + y·sin(θ)
Point-to-line distance: D = |x·cos(θ) + y·sin(θ) − ρ| < τ

AND-logic fusion (3 projections):
  I = S_xy ∩ S_xt ∩ S_yt
  |I| ≥ M_min  →  valid track segment

Rolling-window cost:
  C(T_seg, T_k) = w₁·D_pos + w₂·D_kin
```

---

## 8. References

```
[1]  S. Liu, T. Sun, X. Shu, J. Song, Y. Dong, "Phase-coded FMCW Laser
     Headlamp for ISCAI," IEEE Photonics Technology Letters, 2025.
     DOI: 10.1109/LPT.2025.3649597

[2]  F. Liu et al., "ISAC: Towards dual functional wireless networks for 6G,"
     IEEE J. Sel. Areas Commun., vol. 40, no. 6, pp. 1728–1767, 2022.

[3]  U. Kumbul, N. Petrov, C. S. Vaucher, A. Yarovoy, "Phase-coded FMCW
     for coherent MIMO radar," IEEE Trans. Microw. Theory Tech., vol. 71, 2023.

[4]  U. Kumbul et al., "Smoothed phase-coded FMCW: waveform properties
     and transceiver architecture," IEEE Trans. Aerosp. Electron. Syst., 2023.

[5]  Y. Zhou, J. Liu et al., "A 3D Hough Transform-based TBD technique,"
     Sensors, vol. 19, no. 4, 881, 2019.

[6]  W. Li, W. Yi, K. C. Teh, "Greedy integration based multi-frame detection
     in radar systems," IEEE Trans. Veh. Technol., vol. 72, no. 5, 2023.

[7]  SAE International, "Adaptive Driving Beam (ADB) system performance
     requirements," SAE Standard J3069, Mar. 2021.

[8]  Q. Zheng et al., "A target detection scheme for range-Doppler FMCW
     radar," IEEE Trans. Instrum. Meas., vol. 70, 2021.
```

---

*Research Assignment — Principles of Telecommunication Systems*  
*Democritus University of Thrace | Department of Electrical & Computer Engineering*  
