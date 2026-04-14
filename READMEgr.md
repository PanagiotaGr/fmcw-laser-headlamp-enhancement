# PC-FMCW Laser Headlamp — ISCAI System
## Ερευνητική Εργασία | Αρχές Τηλεπικοινωνιακών Συστημάτων — DUTH

> **Δημοσίευση:** S. Liu, T. Sun, X. Shu, J. Song and Y. Dong,  
> *"Phase-coded FMCW Laser Headlamp for Integrated Sensing, Communication, and Illumination"*  
> IEEE Photonics Technology Letters, DOI: [10.1109/LPT.2025.3649597](https://doi.org/10.1109/LPT.2025.3649597)

---

## Περιεχόμενα

1. [Περίληψη Συστήματος](#1-περίληψη-συστήματος)
2. [Δομή Repository](#2-δομή-repository)
3. [Εγκατάσταση](#3-εγκατάσταση)
4. [Μέρος Α — Αναπαραγωγή Αποτελεσμάτων (6/10)](#4-μέρος-α--αναπαραγωγή-αποτελεσμάτων-610)
5. [Μέρος Β — 19 Ιδέες Βελτίωσης (4/10)](#5-μέρος-β--19-ιδέες-βελτίωσης-410)
6. [Αποτελέσματα & Figures](#6-αποτελέσματα--figures)
7. [Μαθηματικό Υπόβαθρο](#7-μαθηματικό-υπόβαθρο)
8. [Αναφορές](#8-αναφορές)

---

## 1. Περίληψη Συστήματος

Το paper προτείνει ένα **Integrated Sensing, Communication, and Illumination (ISCAI)** σύστημα βασισμένο σε **Phase-coded FMCW (PC-FMCW) laser headlamp** για έξυπνα συνδεδεμένα οχήματα (ICV).

### Τρία Ενοποιημένα Υποσυστήματα

| Υποσύστημα | Τεχνολογία | Επίδοση (paper) |
|---|---|---|
| **Επικοινωνία** | DPSK πάνω σε FMCW phase | 1 Gbps data rate |
| **Sensing (Radar)** | FMCW Range-Doppler + 2D CA-CFAR | 3.8 cm ranging accuracy |
| **Φωτισμός** | Adaptive Driving Beam (ADB) + Phosphor | SAE J3069 compliant |

### Βασικές Παράμετροι

```
fc  = 193.4 THz   (laser carrier, λ ≈ 1551 nm)
B   = 10 GHz      (chirp bandwidth)
T   = 10 μs       (chirp period)
Rb  = 1 Gbps      (data rate)
μ   = B/T         (chirp rate = 10¹⁵ Hz/s)
```

### Αρχιτεκτονική PC-FMCW

```
Εκπεμπόμενο σήμα:
  s_T(t) = A_T · exp{j[2π·fc·t + π·μ·t² + φ_d(t)]}
           ↑ LO chirp           ↑ DPSK data

Ληφθέν echo:
  s_RX(t) = A_R · exp{j[2π·fc·(t-τ) + π·μ·(t-τ)² + φ_d(t-τ) + φ]}

Beat (IF) σήμα:
  s_IF(t) = exp{j[-2π·fb·t + φ_d(t-τ) + φ]}
  όπου fb = μ·τ → εύρος R = c·fb/(2μ)
```

### Αλγόριθμος MHT-TBD

Καινοτομία: **Multidimensional Hough Transform** (4D → 3×2D) με **AND-logic fusion** και **rolling-window** για ελιγμένα targets:

```
3D point cloud (x,y,t)
       ↓
  Προβολή σε 3 επίπεδα: XY, XT, YT
       ↓
  2D HT + 3×3 mean filter σε κάθε επίπεδο
       ↓
  AND-logic: κοινά supporting points → valid tracks
       ↓
  Rolling window: piecewise linear segments
       ↓
  Cost function: C = w₁·D_pos + w₂·D_kin
```

---

## 2. Δομή Repository

```
project/
├── README.md                          ← αυτό το αρχείο
│
├── iscai_reproduction.py              ← Μέρος Α: αναπαραγωγή αποτελεσμάτων
├── iscai_improvement.py               ← Μέρος Β: OFDM + Adaptive MHT (εκτεταμένο)
├── all_19_improvements.py             ← Μέρος Β: και οι 19 ιδέες μαζί
│
├── iscai_results_reproduction.png     ← Figure Μέρους Α (9 subplots)
├── iscai_improvement_proposal.png     ← Figure κύριας πρότασης
└── all_19_improvements.png            ← Figure όλων των ιδεών (5×4 grid)
```

---

## 3. Εγκατάσταση

```bash
pip install numpy scipy matplotlib scikit-learn
```

**Απαιτήσεις:**
- Python ≥ 3.8
- numpy, scipy, matplotlib, scikit-learn

**Εκτέλεση:**
```bash
# Μέρος Α: Αναπαραγωγή
python iscai_reproduction.py

# Μέρος Β: Κύρια πρόταση
python iscai_improvement.py

# Μέρος Β: Όλες οι ιδέες
python all_19_improvements.py
```

---

## 4. Μέρος Α — Αναπαραγωγή Αποτελεσμάτων (6/10)

Αναπαράχθηκαν **6 βασικά αποτελέσματα** από το paper:

### 4.1 CRLB Εύρους (Fig. ~paper eq.7)

Cramér-Rao Lower Bound για εκτίμηση εύρους:

```
var(R̂) ≥ (cT/2B)² · 3/(8π²γMTc²)
σ_R = (c/2)·√var(τ̂)
```

| Παράμετρος | Τιμή |
|---|---|
| SNR (γ) | 10 dB |
| Αριθμός chirps M | 100 |
| CRLB σφάλμα | ~4.6 cm |
| Paper αποτέλεσμα | 3.8 cm |

### 4.2 Range-Doppler Maps (Fig. 2 paper)

- **(a) Καλά διαχωρισμένα targets** (30m/+5m/s, 80m/-8m/s): δύο διακριτές κορυφές
- **(b) Κοντά targets** (50m/+5m/s, 52.5m/+8m/s): επικαλυπτόμενες κορυφές

Υλοποίηση: beat signal generation → GDF phase compensation → 2D FFT → CA-CFAR

### 4.3 MHT-TBD Algorithm (Fig. 4 paper)

**Σενάριο 1:** 2 γραμμικές tracks + Gaussian noise + random clutter  
→ AND-logic fusion, parameter errors: 0.1251 και 0.2348 units

**Σενάριο 2:** 1 γραμμική + 1 μη-γραμμική track σε dense clutter  
→ Rolling-window stitching, mean deviation: **1.6787 units** ✓

### 4.4 ADB Simulation (Fig. 3 paper)

- **Σενάριο ερχόμενου οχήματος:** host 40 km/h, oncoming 30 km/h, αρχική απόσταση 150m
- **Σενάριο πολλαπλών:** host 50 km/h, 2 preceding vehicles με αρχικό spacing 30m

### 4.5 DPSK Communication

Εκτιμώμενο BER = `0.5·exp(-SNR)` για DPSK  
@ SNR=10dB: BER ≈ 2.27×10⁻⁵ → ικανοποιητικό για 1 Gbps

### 4.6 Group Delay Filter (GDF)

```python
# Αποκατάσταση LFM δομής από DPSK phase coding:
H_g(ω) = exp(-j·ω·τ_g(ω))
# Εφαρμογή: FFT(beat) · GDF → IFFT → clean LFM
```

---

## 5. Μέρος Β — 19 Ιδέες Βελτίωσης (4/10)

### Κατηγορία 1: Επικοινωνία

| # | Ιδέα | Βελτίωση | Πολυπλοκότητα |
|---|---|---|---|
| 1 | **OFDM-FMCW** | 2–4× throughput, καλύτερο BER | Μέτρια |
| 2 | **LDPC / Turbo Coding** | +5.5 dB coding gain | Χαμηλή |
| 3 | **Polarization MUX** | 2× throughput (2 Gbps), χωρίς extra BW | Μέτρια |
| 4 | **Adaptive MMSE Equalization** | Robust σε atmospheric turbulence | Μέτρια |

**Κύρια ιδέα (Ιδέα 1 — OFDM):**
- Αντικατάσταση DPSK (1 bit/symbol) με N_sc=64 OFDM υπο-φέρουσες QAM
- Φυσικός διαχωρισμός sensing/comms channels στο frequency domain
- Cyclic prefix αντιμετωπίζει ISI
- BER @ 10dB: DPSK ≈ 6.96×10⁻¹¹ → OFDM-4QAM ≈ 4.04×10⁻¹²

### Κατηγορία 2: Sensing / Radar

| # | Ιδέα | Βελτίωση | Πολυπλοκότητα |
|---|---|---|---|
| 5 | **MUSIC Superresolution** | 10× καλύτερη ανάλυση από FFT | Υψηλή |
| 6 | **Compressed Sensing FMCW** | 70% λιγότερα chirps, ίδια ανάλυση | Μέτρια |
| 7 | **Wideband FMCW (B=50GHz)** | 3.8cm → 1.85cm σφάλμα | Υψηλή HW |
| 8 | **CNN Range-Doppler (NN-CFAR)** | Καλύτερη ROC, robust detection | Υψηλή |

**Κύρια ιδέα (Ιδέα 7 — Wideband):**
```
ΔR = c/(2B):  B=10GHz → ΔR=1.5cm
              B=50GHz → ΔR=0.3cm
σ_R @ SNR=10dB: 3.8cm → 1.85cm
```

### Κατηγορία 3: Tracking / Ανίχνευση

| # | Ιδέα | Βελτίωση | Πολυπλοκότητα |
|---|---|---|---|
| 9 | **Particle Filter TBD** | 1.60u (vs MHT: 3.24u) σε ελιγμένα | Μέτρια |
| 10 | **JPDA** | 80.2% accuracy (vs 64.5% AND-logic) | Μέτρια |
| 11 | **Adaptive MHT Otsu** | FAR: -80%, TDR: 100% | Πολύ χαμηλή |
| 12 | **LSTM Track Prediction** | Καλύτερη απόδοση σε non-linear μανόβρες | Μέτρια |

**Κύρια ιδέα (Ιδέα 11 — Adaptive Otsu):**
```python
# Αντί για fixed threshold:
threshold = 0.7 * accumulator.max()  # paper

# Otsu's method (adaptive):
threshold = otsu_threshold(accumulator)  # πρόταση
# Αποτέλεσμα: FAR 485 → 95 ×10⁻⁴  (-80%)
#             TDR = 1.00 (αναλλοίωτο)
```

### Κατηγορία 4: Φωτισμός ADB

| # | Ιδέα | Βελτίωση | Πολυπλοκότητα |
|---|---|---|---|
| 13 | **Micro-LED Array** | 0.78° vs 3.0° σφάλμα, pixel-level control | Υψηλή HW |
| 14 | **Semantic ADB** | Διαφορετικό safety margin/class | Μέτρια |
| 15 | **LiDAR-guided ADB** | Δy=0, χωρίς camera offset error | Χαμηλή SW |

### Κατηγορία 5: Σύστημα / Waveform Design

| # | Ιδέα | Βελτίωση | Πολυπλοκότητα |
|---|---|---|---|
| 16 | **Cognitive ISAC** | Dynamic sensing/comms split | Μέτρια |
| 17 | **Frequency Hopping FMCW** | P(interf): 90% → 7.6% (5 vehicles) | Χαμηλή |
| 18 | **Waveform Optimization (Pareto)** | Optimal sensing/comms trade-off | Υψηλή |
| 19 | **Optical MIMO Beamforming** | 5 → 32 bits/s/Hz (8×8 MIMO) | Υψηλή HW |

---

## 6. Αποτελέσματα & Figures

### `iscai_results_reproduction.png` — Μέρος Α (9 subplots)

```
┌─────────────────┬─────────────────┬─────────────────┐
│ RDM: καλά διαχ. │ RDM: κοντά tgt  │ CRLB vs SNR     │
├─────────────────┼─────────────────┼─────────────────┤
│ MHT raw data    │ Hough Space XY  │ Track error      │
├─────────────────┼─────────────────┼─────────────────┤
│ ADB: oncoming   │ ADB: multi-veh  │ DPSK BER vs SNR  │
└─────────────────┴─────────────────┴─────────────────┘
```

### `all_19_improvements.png` — Μέρος Β (5×4 grid = 20 subplots)

```
Row 1 [Επικοινωνία]:  BER OFDM | LDPC gain | Pol-MUX throughput | MMSE equalization
Row 2 [Sensing]:      MUSIC vs FFT | CS-FMCW | Wideband B | CNN ROC
Row 3 [Tracking]:     PF trajectory | JPDA accuracy | Otsu MHT | LSTM vs KF
Row 4 [ADB]:          Micro-LED beam | Semantic margins | LiDAR error | ADB summary
Row 5 [Σύστημα]:      Cognitive ISAC | Freq hopping | Pareto optim | MIMO capacity
```

### Συνοπτικός Πίνακας Βελτιώσεων

| Μετρική (paper) | Τιμή paper | Καλύτερη πρόταση | Βελτίωση |
|---|---|---|---|
| Data rate | 1 Gbps | 4 Gbps (OFDM-16QAM) | **4×** |
| Ranging error | 3.8 cm | 1.85 cm (Wideband) | **2×** |
| Track deviation | 1.6787 units | 1.60 units (PF) | **~5%** |
| FAR | — | -80% (Adaptive MHT) | **5×** |
| JPDA accuracy | 64.5% | 80.2% | **+24%** |
| ADB precision | 3.0° | 0.78° (Micro-LED) | **4×** |
| Interference | P≈0.9 | P≈0.076 (Freq. Hop) | **12×** |
| MIMO capacity | 5 b/s/Hz | 32 b/s/Hz (8×8) | **6×** |

---

## 7. Μαθηματικό Υπόβαθρο

### PC-FMCW Σήμα

```
s_LO(t) = A_T · e^{j[2π·fc·t + π·μ·t²]}
s_T(t)  = A_T · e^{j[2π·fc·t + π·μ·t² + φ_d(t)]}   (με DPSK)

φ_d(t) ∈ {0, π}  →  DPSK (differential binary)
Δφ ∈ {0, π}      →  bit=0: χωρίς αλλαγή, bit=1: αλλαγή φάσης
```

### Range-Doppler Processing

```
Beat signal (single target):
  s_IF(t) = e^{j[-2π·fb·t + φ_d(t-τ) + φ]}

2D FFT (range × Doppler):
  X(m,k) = Σ_{n=0}^{N-1} x(m,n) · exp(-j·2π·nk/N)
  V(q,k) = Σ_{m=0}^{M-1} X(m,k) · exp(-j·2π·mq/M)
  Z(q,k) = |V(q,k)|²   (power spectrum)
```

### CRLB (Cramér-Rao Lower Bound)

```
var(τ̂) ≥ (cT/2B)² · 3/(8π²·γ·M·Tc²)
var(v̂) ≥ (λ/2)² · 3/(8π²·γ·Tc²·M³)

όπου: γ = |A|²/σ² = SNR
      Tc = M·T = coherent integration time
```

### ADB Intensity Function

```
ℒ(θ,d) = {  0,                                    θ ∈ Θ, d ≤ d_min
           {  λ·(1-cos(π·(d-d_min)/(d_max-d_min)))/2,  θ ∈ Θ, d_min<d<d_max
           {  1,                                    αλλιώς

Θ = [θ_R - Δy/d - δ, θ_R - Δy/d + δ]   (shadow interval)
```

### MHT Hough Transform

```
Κανονική μορφή γραμμής:  ρ = x·cos(θ) + y·sin(θ)
Απόσταση σημείου:        D = |x·cos(θ) + y·sin(θ) - ρ| < τ

AND-logic fusion:
  I = intersection(S_xy ∩ S_xt ∩ S_yt)
  |I| ≥ M  →  valid track segment
```

---

## 8. Αναφορές

```
[1] Liu et al., "Phase-coded FMCW Laser Headlamp for ISCAI,"
    IEEE Photonics Technology Letters, 2025. DOI: 10.1109/LPT.2025.3649597

[2] Liu et al., "ISAC: Towards dual functional wireless networks for 6G,"
    IEEE JSAC, vol. 40, no. 6, pp. 1728-1767, 2022.

[3] Kumbul et al., "Phase-coded FMCW for coherent MIMO radar,"
    IEEE Trans. Microwave Theory Tech., vol. 71, 2023.

[4] Zhou et al., "A 3D Hough Transform-based TBD technique,"
    Sensors, vol. 19, no. 4, 2019.

[5] SAE International, "Adaptive Driving Beam (ADB) system,"
    SAE Standard J3069, Mar. 2021.

[6] Zheng et al., "Target detection for range-Doppler FMCW radar,"
    IEEE Trans. Instrum. Meas., vol. 70, 2021.
```

---

*Ερευνητική Εργασία — Αρχές Τηλεπικοινωνιακών Συστημάτων*  
*Δημοκρίτειο Πανεπιστήμιο Θράκης | Τμήμα Ηλεκτρολόγων Μηχανικών & Μηχανικών Υπολογιστών*  
