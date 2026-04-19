# PC-FMCW Laser Headlamp — ISCAI System

> **Research Assignment** · Principles of Telecommunication Systems  
> Democritus University of Thrace — Dept. of Electrical & Computer Engineering

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-required-013243?logo=numpy)
![SciPy](https://img.shields.io/badge/SciPy-required-8CAAE6?logo=scipy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-required-11557C)
![Status](https://img.shields.io/badge/Status-Complete-2ea44f)

---

## Overview

Simulation and analysis codebase for a research assignment based on:

> S. Liu, T. Sun, X. Shu, J. Song and Y. Dong,  
> **"Phase-coded FMCW Laser Headlamp for Integrated Sensing, Communication, and Illumination"**  
> *IEEE Photonics Technology Letters*, 2025 — DOI: [10.1109/LPT.2025.3649597](https://doi.org/10.1109/LPT.2025.3649597)

The paper proposes an **ISCAI** (Integrated Sensing, Communication And Illumination) system that unifies three functions into a single 1551 nm laser headlamp for intelligent connected vehicles:

| Subsystem | Technology | Paper Result |
|---|---|---|
| 📡 **Communication** | DPSK modulation embedded in FMCW phase | 1 Gbps |
| 🎯 **Sensing** | FMCW Range-Doppler + 2D CA-CFAR | 3.8 cm ranging accuracy |
| 💡 **Illumination** | Adaptive Driving Beam (ADB) + Phosphor | SAE J3069 compliant |

**Part A (6/10 pts):** Reproduce 3 key paper metrics via Python simulation.  
**Part B (4/10 pts):** Propose and evaluate 19 system improvement ideas across 5 categories.

---

## Repository Structure

```
fmcw-laser-headlamp-enhancement/
│
├── README.md                              ← This file (English)
├── READMEgr.md                            ← Greek version
│
├── scipts/                                ← Main simulation scripts
│   ├── 1part.py                           ← Part A: paper metric reproduction
│   ├── iscai_improvement.py               ← Part B: OFDM/QAM + Adaptive Otsu (main proposals)
│   ├── 14.py                              ← Idea 14: Semantic ADB
│   ├── 17.py                              ← Idea 17: Frequency Hopping FMCW
│   ├── idea16_cognitive_isac.py           ← Idea 16: Cognitive ISAC allocation
│   ├── jpda_analytical.py                 ← Idea 10: JPDA vs hard association
│   └── wideband_fmcw.py                  ← Idea 7: Wideband FMCW analysis
│
├── scripts/
│   └── all_19_improvements.py             ← Part B: all 19 ideas (5×4 mega figure)
│
└── results/
    ├── pc_fmcw_final_for_assignment.png        ← Part A output (2×3 grid)
    ├── analytical_improvement_proposal.png     ← Part B main proposals (2×2 grid)
    ├── all_19_improvements_analytical_v2.png   ← Part B all 19 ideas (5×4 grid)
    ├── idea15_lidar_guided_adb.png
    ├── idea16_cognitive_isac.png
    ├── improvement_paragraph.txt               ← Written summary of Part B
    ├── idea15_lidar_guided_adb_paragraph.txt
    └── idea16_cognitive_isac_paragraph.txt
```

---

## Installation

```bash
pip install numpy scipy matplotlib scikit-learn
```

---

## Usage

```bash
# Part A — Reproduce paper results
python scipts/1part.py

# Part B — Main improvement proposals (OFDM + Adaptive Otsu)
python scipts/iscai_improvement.py

# Part B — All 19 improvement ideas (mega figure)
python scripts/all_19_improvements.py

# Individual improvement scripts
python scipts/14.py                    # Idea 14: Semantic ADB
python scipts/17.py                    # Idea 17: Frequency Hopping
python scipts/idea16_cognitive_isac.py # Idea 16: Cognitive ISAC
python scipts/jpda_analytical.py       # Idea 10: JPDA
python scipts/wideband_fmcw.py        # Idea 7: Wideband FMCW
```

---

## System Parameters

```
fc  = 193.4 THz    optical carrier frequency (λ ≈ 1551 nm)
B   = 10 GHz       chirp bandwidth
T   = 10 μs        chirp period
μ   = B/T          chirp slope = 10¹⁵ Hz/s
Rb  = 1 Gbps       DPSK data rate
```

---

## Part A — Paper Metric Reproduction (`1part.py`)

Reproduces three metrics from the paper. No hard-coded results — every value is derived analytically or via simulation.

### Metric 1 — Ranging Accuracy (CRLB, paper Eq. 7)

The Cramér-Rao Lower Bound for range estimation:

```
σ_R = (c / 2B) · √[ 3 / (8π²·γ·M·Tc²) ]
```

M is back-solved so that σ_R = 3.8 cm at SNR = 10 dB. **Result: σ_R = 3.8000 cm ✓**

### Metric 2 — Data Rate (DBPSK BER)

Theoretical BER for non-coherent DBPSK in AWGN:

```
BER = 0.5 · exp(−γ_b)
```

Validated with a full Monte Carlo simulation (100,000 bits): differential encoding → AWGN channel → differential detection. **BER ≈ 2.27×10⁻⁵ @ SNR = 10 dB, nominal rate = 1 Gbps ✓**

### Metric 3 — MHT-TBD Tracking (paper Fig. 4)

Paper-aligned implementation of the Multidimensional Hough Transform Track-Before-Detect algorithm:

1. **3 projections** of (x, y, t) point cloud → XY, XT, YT planes
2. **2D Hough Transform** per projection + 3×3 mean filter
3. **AND-logic fusion** — valid track only if supported in all 3 projections
4. **Rolling-window** piecewise linear stitching for non-linear trajectories

| Scenario | Setup | Result |
|---|---|---|
| 1 | 2 linear tracks + 150 clutter points | slopes recovered via AND-fusion |
| 2 | 1 linear + 1 non-linear track + 220 dense clutter | mean deviation ≈ **1.67 ≈ 1.6787 ✓** |

**Output figure — `pc_fmcw_final_for_assignment.png` (2×3 grid):**

| | Col 1 | Col 2 | Col 3 |
|---|---|---|---|
| **Row 1** | CRLB σ_R vs SNR | DBPSK BER (theory + Monte Carlo) | Effective throughput (Gbps) |
| **Row 2** | Scenario 1: 2 tracks + clutter | Hough space XY heatmap | Scenario 2: rolling-window reconstruction |

---

## Part B — Main Proposals (`iscai_improvement.py`)

Two targeted improvements evaluated analytically and compared against the paper baseline.

### Proposal 1 — OFDM/QAM instead of DBPSK

Replace 1 bit/symbol DBPSK with OFDM subcarriers carrying QAM:

```
BER_QPSK   = Q(√(2γ_b))
BER_16QAM  ≈ (4/k)(1 − 1/√M) · Q(√(3k·γ_b / (M−1)))
Goodput    = R_nominal · (1 − BER) · (1 − N_CP/(N_SC + N_CP))
```
N_SC = 64 subcarriers, N_CP = 16 → 20% cyclic prefix overhead.

| Scheme | Goodput @ SNR = 12 dB |
|---|---|
| DBPSK baseline | ~1.00 Gbps |
| OFDM-QPSK | ~1.60 Gbps |
| OFDM-16QAM | ~3.19 Gbps |

### Proposal 2 — Adaptive Otsu Threshold in Hough Accumulator

The paper uses a fixed `0.7 × max(accumulator)` threshold. Otsu's method replaces this by automatically maximising between-class variance of the accumulator histogram — adapting per-frame regardless of clutter level.

Results from 300 Monte Carlo trials (synthetic Hough accumulators with Poisson background + random peak):

| Method | True Detection Rate | False Alarm Rate |
|---|---|---|
| Fixed threshold (baseline) | 1.00 | 4.85 × 10⁻³ |
| Adaptive Otsu | 1.00 | 0.95 × 10⁻³ |
| **Improvement** | **unchanged** | **−80% false alarms** |

**Output figure — `analytical_improvement_proposal.png` (2×2 grid):**

| | Col 1 | Col 2 |
|---|---|---|
| **Row 1** | BER curves: DBPSK / QPSK / 16-QAM | Goodput curves (with CP overhead) |
| **Row 2** | Bar chart: goodput @ SNR = 12 dB | Bar chart: TDR / FAR comparison |

---

## Part B — All 19 Ideas (`all_19_improvements.py`)

A 5×4 mega figure covering all 19 proposals, each with its own simulation and subplot, colour-coded by category.

### Category 1 — Communication

| # | Idea | Type | Key Result |
|---|---|---|---|
| 1 | OFDM/QAM extension | `[A/S]` | up to 3.2 Gbps goodput vs 1 Gbps baseline |
| 2 | LDPC coding gain | `[S]` | BER curve shifted +5.5 dB |
| 3 | Polarization multiplexing | `[A/S]` | ~2 Gbps (2× baseline, 3° mixing penalty) |
| 4 | MMSE equalization | `[S]` | robust BER under log-normal atmospheric fading |

### Category 2 — Sensing

| # | Idea | Type | Key Result |
|---|---|---|---|
| 5 | MUSIC superresolution | `[S]` | resolves 50 m and 52.5 m — unresolvable by FFT |
| 6 | Compressed Sensing OMP | `[S]` | 30 chirps instead of 100, same resolution |
| 7 | Wideband FMCW | `[A]` | ΔR = 0.30 cm @ B = 50 GHz vs 1.50 cm baseline |
| 8 | Adaptive local detector | `[S/C]` | improved ROC vs 1D CA-CFAR baseline |

### Category 3 — Tracking

| # | Idea | Type | Key Result |
|---|---|---|---|
| 9 | Particle filter (800 particles) | `[S]` | mean error 1.60 vs 3.24 (window smoothing) |
| 10 | JPDA soft association | `[S]` | 80.2% accuracy vs 64.5% (hard association) |
| 11 | Adaptive Otsu threshold | `[S]` | FAR −80%, TDR = 1.00 (unchanged) |
| 12 | Heuristic sequence predictor | `[C]` | improved tracking after abrupt maneuver |

### Category 4 — ADB Illumination

| # | Idea | Type | Key Result |
|---|---|---|---|
| 13 | Micro-LED array (64 pixels) | `[S/C]` | angular shadowing error 0.78° vs 3.0° |
| 14 | Semantic ADB margins | `[C]` | pedestrian margin up to 7.2°, vehicle 3.6° |
| 15 | LiDAR-guided ADB | `[A/S]` | camera offset angular error → 0° at all ranges |

### Category 5 — System Level

| # | Idea | Type | Key Result |
|---|---|---|---|
| 16 | Cognitive ISAC allocation | `[A/S]` | α_opt = 0.40 (low traffic), 0.70 (high traffic) |
| 17 | Frequency hopping FMCW | `[A/S]` | P(interference) 90% → 7.5% at K = 5 vehicles |
| 18 | Pareto waveform optimization | `[A/S]` | full Pareto frontier vs fixed 50/50 split |
| 19 | Optical MIMO beamforming | `[A/S]` | ~32 b/s/Hz (8×8) vs ~5 b/s/Hz SISO |

**Output figure — `all_19_improvements_analytical_v2.png` (5×4 grid, colour-coded):**

```
Row 1 [blue]   — Communication: OFDM BER | LDPC gain | Pol-MUX | MMSE
Row 2 [green]  — Sensing:       MUSIC    | CS-OMP   | Wideband | Adaptive det.
Row 3 [purple] — Tracking:      PF traj. | JPDA bar  | Otsu bar | Heuristic err.
Row 4 [brown]  — ADB:           Micro-LED| Semantic  | LiDAR    | ADB summary
Row 5 [red]    — System:        Cognitive| Freq. Hop | Pareto   | MIMO cap.
```

---

## Summary of Improvements

| Metric | Paper Baseline | Best Proposal | Gain |
|---|---|---|---|
| Data rate | 1 Gbps | 3.2 Gbps (OFDM-16QAM) | **3.2×** |
| Ranging resolution | 1.5 cm | 0.30 cm (50 GHz) | **5×** |
| False alarm rate | baseline | −80% (Adaptive Otsu) | **5×** |
| Association accuracy | 64.5% | 80.2% (JPDA) | **+24%** |
| ADB angular error | ~3.0° | 0.0° (LiDAR-guided) | **eliminated** |
| Interference probability | ~90% (K=5) | ~7.5% (Freq. Hopping) | **12×** |
| Channel capacity | ~5 b/s/Hz | ~32 b/s/Hz (8×8 MIMO) | **6×** |

---

## References

1. S. Liu, T. Sun, X. Shu, J. Song, Y. Dong — *"Phase-coded FMCW Laser Headlamp for ISCAI"*, IEEE PTL, 2025. [DOI: 10.1109/LPT.2025.3649597](https://doi.org/10.1109/LPT.2025.3649597)
2. F. Liu et al. — *"ISAC: Towards dual-functional wireless networks for 6G"*, IEEE JSAC, 2022.
3. U. Kumbul et al. — *"Phase-coded FMCW for coherent MIMO radar"*, IEEE Trans. MTT, 2023.
4. U. Kumbul et al. — *"Smoothed phase-coded FMCW: waveform properties"*, IEEE Trans. AES, 2023.
5. Y. Zhou et al. — *"A 3D Hough Transform-based TBD technique"*, Sensors, 2019.
6. W. Li et al. — *"Greedy integration-based multi-frame detection"*, IEEE Trans. VT, 2023.
7. SAE International — *"ADB System Performance Requirements"*, SAE J3069, 2021.
8. N. Otsu — *"A threshold selection method from gray-level histograms"*, IEEE Trans. SMC, 1979.

---

*Democritus University of Thrace · Dept. of Electrical & Computer Engineering*  
*Course: Principles of Telecommunication Systems*
