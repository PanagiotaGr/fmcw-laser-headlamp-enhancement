# ISCAI Extensions — 30 Ιδέες Βελτίωσης
## Επεκτάσεις του PC-FMCW Laser ISCAI Συστήματος

> Βασίζεται στο paper: Liu et al., *"Phase-coded FMCW Laser Headlamp for ISCAI"*, IEEE PTL, 2025  
> Δημοκρίτειο Πανεπιστήμιο Θράκης — Αρχές Τηλεπικοινωνιακών Συστημάτων

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python)
![Status](https://img.shields.io/badge/Simulations-Ολοκληρώθηκαν-2ea44f)
![Ideas](https://img.shields.io/badge/Ιδέες-30-185FA5)

---

## Πίνακας Περιεχομένων

1. [Επισκόπηση](#1-επισκόπηση)
2. [Αρχεία & Εκτέλεση](#2-αρχεία--εκτέλεση)
3. [Κατηγορία 1 — Cooperative Sensing (Ιδέες 1–5)](#3-κατηγορία-1--cooperative-sensing)
4. [Κατηγορία 2 — Adaptive Headlamp (Ιδέες 6–10)](#4-κατηγορία-2--adaptive-headlamp)
5. [Κατηγορία 3 — Πεζοί & Cyclists (Ιδέες 11–16)](#5-κατηγορία-3--πεζοί--cyclists)
6. [Κατηγορία 4 — Robustness (Ιδέες 17–20)](#6-κατηγορία-4--robustness)
7. [Κατηγορία 5 — AI & Future (Ιδέες 21–30)](#7-κατηγορία-5--ai--future)
8. [Συγκεντρωτικά Αποτελέσματα](#8-συγκεντρωτικά-αποτελέσματα)

---

## 1. Επισκόπηση

Το baseline ISCAI σύστημα (Liu et al., 2025) λειτουργεί με **ένα αυτοκίνητο**, **σταθερές παραμέτρους**, και **απλά ανθρωποειδή αντικείμενα** (vehicles). Οι 30 ιδέες επεκτείνουν το σύστημα σε:

- **Συνεργατική λειτουργία** πολλών αυτοκινήτων
- **Προσαρμοστική σάρωση** ανάλογα με κίνδυνο/καιρό
- **Ανίχνευση & πρόβλεψη** πεζών και ποδηλατών
- **Ανθεκτικότητα** σε παρεμβολές και κακές καιρικές συνθήκες
- **AI-based** βελτιστοποίηση σε πραγματικό χρόνο

### Βαθμίδες Αξιοπιστίας

| Σύμβολο | Σημασία |
|---|---|
| `[A]` | Αναλυτικό — κλειστός τύπος, πλήρης μαθηματική απόδειξη |
| `[A/S]` | Αναλυτικό + Simulation |
| `[S]` | Semi-analytical simulation |
| `[C]` | Conceptual — proof-of-concept, future work |

---

## 2. Αρχεία & Εκτέλεση

```
iscai_extensions/
├── idea01_virtual_giant_sensor.py    ← Ιδέα 1: SAR με 2 αμάξια
├── idea03_blind_spot_coverage.py     ← Ιδέα 3: Τυφλές γωνίες
├── ideas_06_11_12_17.py              ← Ιδέες 6, 11, 12, 17 (top picks)
├── ideas_remaining_mega.py           ← Ιδέες 2,4,5,7-10,13-16,18-20,22-30
│
├── idea01_virtual_giant_sensor.png   ← Εικόνα ιδέας 1
├── idea03_blind_spot_coverage.png    ← Εικόνα ιδέας 3
├── ideas_06_11_12_17.png             ← Εικόνα ιδεών 6,11,12,17 (2×4 grid)
└── ideas_remaining_mega.png          ← Mega figure 5×5 grid
```

### Εκτέλεση

```bash
python idea01_virtual_giant_sensor.py
python idea03_blind_spot_coverage.py
python ideas_06_11_12_17.py
python ideas_remaining_mega.py
```

---

## 3. Κατηγορία 1 — Cooperative Sensing

Ιδέες όπου **πολλά αυτοκίνητα συνεργάζονται** για καλύτερη αίσθηση.

---

### Ιδέα 1 — Δύο Αμάξια ως Virtual Giant Sensor `[A/S]` ⭐ TOP PICK

**Αρχείο:** `idea01_virtual_giant_sensor.py`

#### Το Πρόβλημα
Ένα μόνο αμάξι έχει μικρό aperture (D ≈ 5 cm). Η angular resolution είναι θ = λ/D — πολύ περιορισμένη για διαχωρισμό κοντινών στόχων σε cross-range.

#### Η Ιδέα — Synthetic Aperture
Δύο αυτοκίνητα σε απόσταση B = 2m μεταξύ τους στέλνουν και λαμβάνουν coherently. Το αποτέλεσμα είναι ένα **virtual aperture = B**, σαν να έχουν ένα τεράστιο sensor:

```
Virtual aperture cross-range resolution:
Δx_virtual = R × λ / B

Βελτίωση έναντι single car:
Gain = B / D = 2 m / 0.05 m = 40×

SNR κέρδος (coherent sum):
ΔSNRdB = 10 log₁₀(2) = +3.0 dB
```

#### Τι Κάνει ο Κώδικας
1. Δημιουργεί δύο στόχους στα 50m και 52.5m (cross-range separation = 0.6m)
2. Παράγει beat signal M×N matrix για κάθε αμάξι ξεχωριστά
3. Coherent combination: `R_virtual = R_carA + R_carB`
4. 2D FFT → Range-Doppler map για single vs virtual
5. Cross-range profile σύγκριση στο target range

#### Αποτελέσματα

| Μετρική | Single Car | Virtual Sensor | Βελτίωση |
|---|---|---|---|
| Cross-range resolution | 1.24 mm | 0.031 mm | **40×** |
| SNR | baseline | +3.0 dB | **+3 dB** |
| Διαχωρισμός στόχων @40m | αδύνατο | εφικτό | ✓ |

---

### Ιδέα 2 — Cooperative Area Splitting `[S]`

**Υλοποίηση:** `ideas_remaining_mega.py`

#### Η Ιδέα
Αντί να σκανάρουν και τα δύο αμάξια την ίδια περιοχή, χωρίζουν τον χώρο σε sectors:
- Αμάξι Α → αριστερά
- Αμάξι Β → δεξιά

```python
T_scan_single = 1.0 s   # full scan 1 αμάξι
T_scan_coop   = 0.5 s   # full scan 2 αμάξια (κάθε ένα μισό)
```

#### Αποτέλεσμα
- Χρόνος κάλυψης: **2× ταχύτερο**
- Temporal resolution: **2×** καλύτερο
- Ιδανικό για: platooning, intersection monitoring

---

### Ιδέα 3 — Blind Spot Coverage `[A/S]` ⭐ TOP PICK

**Αρχείο:** `idea03_blind_spot_coverage.py`

#### Το Πρόβλημα
Ένα σταθμευμένο φορτηγό (8m × 2.5m) κρύβει έναν πεζό πίσω του. Το αμάξι Α δεν μπορεί να δει τον πεζό λόγω occlusion.

#### Η Ιδέα
Αμάξι Β βρίσκεται σε παράλληλη λωρίδα (3.5m πλάι). Λόγω της διαφορετικής γωνίας παρατήρησης, βλέπει τον πεζό και στέλνει warning μέσω V2X.

#### Μοντέλο Occlusion
```python
# Line-of-sight check:
# Αν εμπόδιο βρίσκεται μεταξύ sensor και target:
if sx < obstacle_x and tx > obstacle_x:
    y_at_obstacle = sy + t * (ty - sy)  # interpolation
    if obstacle_y_min ≤ y_at_obstacle ≤ obstacle_y_max:
        return occluded = True
```

#### Αποτελέσματα

| Μετρική | Μόνο Αμάξι Α | Με Β (Cooperative) | Βελτίωση |
|---|---|---|---|
| Χρόνος ανίχνευσης | t = 1.00 s | t = 0.40 s | **0.60 s νωρίτερα** |
| TTC (Time-To-Collision) | 3.25 s | 3.85 s | **+0.60 s** |
| Πεζός ορατός | Μετά το truck | Πριν το truck | ✓ |

> **Γιατί +0.60s είναι σημαντικό;** Μελέτες δείχνουν ότι ο μέσος χρόνος αντίδρασης οδηγού είναι 0.7–1.5s. Το +0.60s μπορεί να κάνει τη διαφορά μεταξύ σύγκρουσης και αποφυγής.

---

### Ιδέα 4 — Cooperative Pedestrian Tracking `[S]`

**Υλοποίηση:** `ideas_remaining_mega.py`

Bayesian sensor fusion: Ν αμάξια μετρούν τον ίδιο πεζό ανεξάρτητα, το αποτέλεσμα συνδυάζεται ως weighted average.

| Αριθμός Sensors | Position Error | Βελτίωση |
|---|---|---|
| 1 sensor | 0.610 m | baseline |
| 2 sensors | 0.436 m | −29% |
| 4 sensors | 0.311 m | **−49%** |

---

### Ιδέα 5 — Platoon Sensing `[A/S]`

Σε πλατούν N αυτοκινήτων, το SNR βελτιώνεται λόγω consensus averaging:

```
SNR_gain = 10 × log₁₀(N)

N = 2:  +3.0 dB
N = 4:  +6.0 dB
N = 8:  +9.0 dB  ← σχεδόν 10 dB κέρδος!
```

---

## 4. Κατηγορία 2 — Adaptive Headlamp

Ιδέες όπου το σύστημα **αλλάζει παραμέτρους δυναμικά** ανάλογα με την κατάσταση.

---

### Ιδέα 6 — Danger-Adaptive Scan Mode `[A/S]` ⭐ TOP PICK

**Υλοποίηση:** `ideas_06_11_12_17.py`

#### Η Ιδέα
Αντί για σταθερές παραμέτρους (B, T, M), το σύστημα λειτουργεί σε τρεις λειτουργίες ανάλογα με το επίπεδο κινδύνου:

```
State machine:
LOW danger  → B = 5 GHz,  T = 20 μs, M = 20   (εξοικονόμηση ενέργειας)
MED danger  → B = 10 GHz, T = 10 μs, M = 50   (baseline paper)
HIGH danger → B = 20 GHz, T = 5 μs,  M = 100  (μέγιστη ακρίβεια)
```

#### Αποτελέσματα @ SNR = 10 dB

| Mode | B | M | σ_R (cm) | Σχετική Ισχύς |
|---|---|---|---|---|
| LOW danger | 5 GHz | 20 | 103.4 cm | −6 dBr |
| MED danger (baseline) | 10 GHz | 50 | 26.2 cm | 0 dBr |
| HIGH danger | 20 GHz | 100 | **9.2 cm** | +6 dBr |

> Το HIGH mode δίνει **3× καλύτερη ακρίβεια** από το baseline, σε κρίσιμες καταστάσεις.

---

### Ιδέα 7 — Adaptive Beam Steering `[S]`

Ο laser προσανατολίζεται προς τη ζώνη όπου εκτιμάται κίνδυνος. Το SNR μειώνεται εκτός της κεντρικής γωνίας:

```
SNR(θ) = SNR_max - 0.3 × |θ - θ_target|^1.5
```

Αποτέλεσμα: Εξοικονόμηση ισχύος έως 40% σε ήρεμα σενάρια.

---

### Ιδέα 8 — Hazard-Priority Illumination `[S]`

Αναλογική κατανομή ισχύος φωτισμού βάσει risk score ανά ζώνη:

```python
power_alloc[zone] = risk_score[zone] / sum(risk_scores)
```

| Ζώνη | Risk Score | Power Allocation |
|---|---|---|
| Straight road | 0.20 | 8.5% |
| Left curve | 0.50 | 21.3% |
| Junction | 0.80 | 34.0% |
| Blind spot | 1.00 | **42.6%** |

---

### Ιδέα 9 — Pedestrian-Priority Headlamp `[S]`

Διαφορετικό safety margin ανά κατηγορία αντικειμένου, με priority weighting:

```python
final_margin = base_margin × priority_weight / min(weights)
```

| Αντικείμενο | Base Margin | Priority | Final Margin |
|---|---|---|---|
| Όχημα | 2.0° | 1.0× | 2.0° |
| Ποδηλάτης | 3.5° | 1.5× | 5.25° |
| Πεζός | 4.0° | 2.0× | 8.0° |
| Παιδί | 5.0° | 3.0× | **15.0°** |

---

### Ιδέα 10 — Traffic-Adaptive Phase Codes `[A/S]`

Δυναμική αλλαγή Rb ανάλογα με πυκνότητα κυκλοφορίας:

```
Rb(traffic) = Rb_max × (1 - 0.5 × traffic_density)
```

- Χαμηλή κυκλοφορία → 1 Gbps (full rate)
- Πυκνή κυκλοφορία → 0.5 Gbps (robust mode, καλύτερο BER)

---

## 5. Κατηγορία 3 — Πεζοί & Cyclists

---

### Ιδέα 11 — Pedestrian Trajectory Prediction `[S]` ⭐ TOP PICK

**Υλοποίηση:** `ideas_06_11_12_17.py`

#### Αλγόριθμος — Social Force Model (Helbing & Molnar, 1995)

Κάθε πεζός δέχεται τρεις δυνάμεις:

```
F_total = F_desired + F_social + F_obstacle

F_desired = (mass/τ) × (v₀ × ê_destination - v_current)
            ↑ "θέλω να πάω εκεί με ταχύτητα v₀"

F_social  = A × exp(-d/B) × ê_away
            ↑ "απωθούμαι από άλλους πεζούς"

τ  = 0.5 s    (relaxation time)
v₀ = 1.3 m/s  (desired speed)
A  = 2000 N   (social force amplitude)
B  = 0.08 m   (social force range)
```

#### Prediction Pipeline
```
1. Παρατήρηση τρέχουσας θέσης & ταχύτητας
2. SFM integration για horizon δευτερόλεπτα
3. Παράγωγη predicted trajectory
4. Σύγκριση με ground truth → ADE (Average Displacement Error)
```

#### Αποτελέσματα

| Prediction Horizon | ADE |
|---|---|
| 0.5 s | ~0.00 m (excellent) |
| 1.0 s | ~0.00 m |
| 2.0 s | ~0.00 m |

> **Σημείωση:** Τα μηδενικά ADE οφείλονται στο ότι το SFM είναι ντετερμινιστικό όταν δεν υπάρχει εξωτερική διαταραχή. Σε πραγματικά δεδομένα (noise + εμπόδια) το ADE αυξάνεται σε 0.3–1.5m για 2s horizon.

---

### Ιδέα 12 — Crossing Intention Prediction `[S]` ⭐ TOP PICK

**Υλοποίηση:** `ideas_06_11_12_17.py`

#### Το Πρόβλημα
Πότε ένας πεζός θα περάσει τον δρόμο; Πρέπει να ξέρουμε 1-2s νωρίτερα.

#### Features & Classifier

```
Features:
  x₁ = heading_angle / 360       (κατεύθυνση προς δρόμο)
  x₂ = speed / 3                 (ταχύτητα)
  x₃ = dist_to_curb / 10         (απόσταση από πεζοδρόμιο)
  x₄ = lateral_velocity / 2      (συνιστώσα προς δρόμο)

Logistic Regression:
  P(crossing) = σ(w₁x₁ + w₂x₂ + w₃x₃ + w₄x₄ + b)
  σ(z) = 1 / (1 + e⁻ᶻ)

Βέλτιστα βάρη (μετά 500 epochs):
  w = [+2.996, +1.199, -2.726, -1.141]
      ↑ heading  ↑ speed   ↑ far=safe   ↑ low lat_v=safe
```

#### Αποτελέσματα (600 samples, 80/20 split)

| Μετρική | Τιμή |
|---|---|
| Test accuracy | **100.0%** |
| AUC-ROC | **1.000** |
| Πιο σημαντικό feature | Heading angle |
| Δεύτερο σημαντικό | Distance to curb |

---

### Ιδέα 13 — Hidden Pedestrian (Shadow Sensing) `[C]`

Έμμεση ανίχνευση πεζού από τη "σκιά" που αφήνει στο Range-Doppler map.

```
P(detection via shadow) = 1 - exp(-SNR_shadow / 5)

SNR_shadow ≈ 15 - 20×log₁₀(R/5) - 10  [dB]
```

Εφικτό σε R < 20m με SNR > 5 dB.

---

### Ιδέα 14 — Child / Erratic Motion Detection `[S]`

Ανίχνευση παιδιών από το **jerk** (d³x/dt³) της τροχιάς τους:

```
Jerk = d³x/dt³

Mean jerk adult:  157.95 m/s³
Mean jerk child:  831.19 m/s³  ← 5.3× υψηλότερο!

Alert trigger: jerk > 2 × mean_adult_jerk
```

---

### Ιδέα 15 — Cyclist Trajectory Prediction `[S]`

Ξεχωριστό κινηματικό μοντέλο για ποδηλάτες (unicycle dynamics):

```
Ποδηλάτης: v ≈ 4-6 m/s, μικρό lateral displacement
Πεζός:     v ≈ 1-2 m/s, μεγαλύτερο lateral displacement
```

Σύγκριση Particle Filter vs Kalman Filter για tracking στροφών.

---

### Ιδέα 16 — NLOS Detection `[C]`

Ανίχνευση πεζού μέσω Non-Line-of-Sight reflections (από τοίχο ή έδαφος):

```
P(detect via NLOS) = 1 - exp(-γ_NLOS / 5)
γ_NLOS = SNR_NLOS(R)  →  φθίνει γρήγορα με R
```

Εφικτό σε R < 15m.

---

## 6. Κατηγορία 4 — Robustness

---

### Ιδέα 17 — Self-Healing Phase Codes `[A/S]` ⭐ TOP PICK

**Υλοποίηση:** `ideas_06_11_12_17.py`

#### Το Πρόβλημα
Βροχή και ομίχλη προκαλούν optical attenuation που μειώνει το SNR:

```
Rain attenuation (ITU-R P.838):
  α(R) = k × R^a     [dB/km]
  k = 0.002, a = 1.0  για λ = 1550 nm

Fog attenuation (Kim model):
  β = 3.91/V × (λ/0.55)^(-q)   [dB/km]
  V = visibility (km), q = f(V)
```

#### Adaptive Mode Switching

```
if SNR > 12 dB:   → CLEAR mode   (1 Gbps, B=10 GHz, M=50)
elif SNR > 5 dB:  → RAIN mode    (0.5 Gbps, T=20μs, M=100)
else:             → FOG mode     (0.25 Gbps, B=5 GHz, M=200)
```

#### Αποτελέσματα @ 50m range

| Καιρός | Attenuation | SNR | Mode | BER |
|---|---|---|---|---|
| Clear | 0.00 dB | 12.0 dB | RAIN | 6.5 × 10⁻⁸ |
| Light rain (5 mm/h) | 0.00 dB | 12.0 dB | RAIN | 6.6 × 10⁻⁸ |
| Heavy rain (50 mm/h) | 0.02 dB | 12.0 dB | RAIN | 7.1 × 10⁻⁸ |
| Dense fog (V=100m) | 0.51 dB | 11.5 dB | RAIN | 3.8 × 10⁻⁷ |

---

### Ιδέα 18 — Fog/Rain Optimized Sensing `[A/S]`

Βέλτιστη επιλογή (B, T) ανά καιρικές συνθήκες. Μεγαλύτερο T βοηθά σε ομίχλη λόγω coherent integration gain:

| B (GHz) | σ_R (cm) in fog |
|---|---|
| 2 | υψηλό |
| 10 | baseline |
| 50 | **0.30 cm** |

---

### Ιδέα 19 — Energy-Efficient ISAC `[A/S]`

Sparse chirp scheduling: αντί για συνεχή εκπομπή, εκπέμπω μόνο x% των frames:

```
σ_R(duty) = σ_R_baseline / √(duty_cycle)
```

Pareto frontier: ενέργεια vs ακρίβεια. Στο 50% duty cycle: 41% εξοικονόμηση ενέργειας, √2 = 41% χειρότερη ακρίβεια.

---

### Ιδέα 20 — Anti-Interference Coding `[A/S]`

Extension του Frequency Hopping με coordinated CDMA-style code assignment:

```
P(collision, K vehicles, N codes) = 1 - (1 - 1/N)^K

K=5 αμάξια:
  N=8:    P = 48.6%
  N=16:   P = 28.2%
  N=32:   P = 15.2%
  N=64:   P = 7.6%
  N=128:  P = 3.8%
```

---

## 7. Κατηγορία 5 — AI & Future

---

### Ιδέα 21 — AI-Generated Phase Codes `[C]`

PyTorch autoencoder που μαθαίνει βέλτιστους phase codes:
- Encoder: bits → phase sequence
- Channel: FMCW model
- Decoder: received → bits
- Joint optimization: minimize BER + maximize sensing SINR

---

### Ιδέα 22 — Reinforcement Learning Sensing `[C]`

Q-learning agent επιλέγει (B, T, M) ανά frame:

```python
States:   10 SNR levels
Actions:  LOW/MED/HIGH bandwidth
Reward:   -CRLB(B,M) - energy_cost × action
```

**Αποτέλεσμα:** Συγκλίνει σε ~160 episodes. Μετά τη σύγκλιση: βέλτιστο trade-off ακρίβεια/ενέργεια.

---

### Ιδέα 23 — Digital Twin Guided Sensing `[C]`

Risk map του δρόμου (occupancy grid) καθοδηγεί πού να σκανάρει πρώτα:

```
scan_priority[x,y] = risk_map[x,y] / sum(risk_map)
```

High-risk zones (σταυροδρόμια, νησίδες) σκανάρονται συχνότερα.

---

### Ιδέα 24 — Game Theory Cooperative Sensing `[C]`

2-player game: κάθε αμάξι επιλέγει sensing effort α ∈ [0,1].

```
Coverage = 1 - (1 - α_A)(1 - α_B)
Nash equilibrium: (α_A, α_B) = (1, 1) [dominant strategy]
```

Με mechanism design: βέλτιστο αποτέλεσμα με λιγότερη συνολική ενέργεια.

---

### Ιδέα 25 — Multi-Objective ISAC Optimization `[A/S]`

Ταυτόχρονη βελτιστοποίηση 3 στόχων:

```
Παράμετρος α ∈ [0,1]:
  SINR_sensing    = 10 log₁₀(α × γ × M)
  R_communication = log₂(1 + (1-α) × γ)
  Illumination    = 1 - α  (relative power)
```

Pareto frontier σε 3D χώρο. Κάθε σημείο = βέλτιστος συμβιβασμός.

---

### Ιδέα 26 — Communicative Headlamp Patterns `[C]`

Ο προβολέας χρησιμοποιεί flash patterns για να επικοινωνεί με πεζούς:

| Pattern | Σήμα |
|---|---|
| `1-0-1-0-1-0` | STOP |
| `1-1-0-1-1-0` | CROSSING SAFE |
| `1-0-0-1-0-0` | CAUTION |
| `1-1-1-0-0-0` | ALL CLEAR |

Information rate: ~2 bits/pattern @ 2 patterns/second = ~4 bits/s.

---

### Ιδέα 27 — Distributed LiDAR Network `[C]`

N αυτοκίνητα σαν distributed sensor network:

```
Coverage(N)    = 1 - exp(-0.4×N)
Map quality(N) = 1 - 1/(2N)

N=1: coverage 33%, quality 50%
N=4: coverage 80%, quality 88%
N=7: coverage 94%, quality 93%
```

---

### Ιδέα 28 — TTC-Adaptive Waveform `[A/S]`

Bandwidth αυξάνεται αυτόματα όταν TTC μικραίνει:

```
B(TTC) = B₀ × (1 + 5 / (TTC + 0.5))

TTC = 5.0 s: B = 10 GHz   (baseline)
TTC = 1.0 s: B = 35 GHz   (3.5×)
TTC = 0.5 s: B = 60 GHz   (6×!)
```

Σε κρίσιμη κατάσταση (TTC = 0.5s): σ_R μειώνεται 6× αυτόματα.

---

### Ιδέα 29 — Uncertainty-Driven Sensing `[C]`

Active sensing: σκανάρω μόνο εκεί που η αβεβαιότητα είναι μεγάλη.

```
Entropy map: H(x,y) = -p×log₂(p) - (1-p)×log₂(1-p)
Scan priority = H(x,y) / Σ H

Αποτέλεσμα: Ίδια πληροφορία με λιγότερες μετρήσεις
```

---

### Ιδέα 30 — Closed-Loop ISAC `[S]`

Full feedback loop: κάθε υποσύστημα τροφοδοτεί το επόμενο:

```
Sensing → Tracking → Waveform Opt. → Illumination → [feedback] → Sensing
```

**Αποτέλεσμα:** Και τα τρία υποσυστήματα συγκλίνουν σε ~12 iterations:
- Sensing quality: 0.30 → 1.00
- Tracking quality: 0.30 → 1.00
- Illumination accuracy: 0.20 → 1.00

---

## 8. Συγκεντρωτικά Αποτελέσματα

### Κορυφαία Αποτελέσματα ανά Κατηγορία

| # | Ιδέα | Τύπος | Κύριο Αποτέλεσμα |
|---|---|---|---|
| 1 | Virtual Giant Sensor | `[A/S]` | Cross-range resolution **40×** |
| 3 | Blind Spot Coverage | `[A/S]` | Ανίχνευση **0.6s νωρίτερα** |
| 5 | Platoon Sensing | `[A]` | SNR κέρδος **+9 dB** (N=8) |
| 6 | Danger-Adaptive Mode | `[A/S]` | σ_R **9.2 cm** (HIGH mode) |
| 9 | Pedestrian Priority | `[S]` | Child margin **15°** vs 2° |
| 12 | Crossing Intention | `[S]` | Accuracy **100%**, AUC **1.0** |
| 14 | Child Detection | `[S]` | Jerk ratio child/adult **5.3×** |
| 17 | Self-Healing Codes | `[A/S]` | Auto mode switch σε fog/rain |
| 20 | Anti-Interference | `[A]` | P(collision) **3.8%** (N=128) |
| 22 | RL Sensing | `[C]` | Σύγκλιση σε **~160 episodes** |
| 28 | TTC-Adaptive | `[A/S]` | B **6×** @ TTC = 0.5s |
| 30 | Closed-Loop ISAC | `[S]` | Σύγκλιση σε **12 iterations** |

#
---

## Βιβλιογραφία

1. S. Liu et al. — *"Phase-coded FMCW Laser Headlamp for ISCAI"*, IEEE PTL, 2025.
2. D. Helbing, P. Molnar — *"Social force model for pedestrian dynamics"*, PRE, 1995.
3. ITU-R P.838 — *"Specific attenuation model for rain"*, 2005.
4. M. Kim — *"Visibility of fog"*, Optical Engineering, 2001.
5. F. Liu et al. — *"ISAC: Towards dual-functional wireless networks for 6G"*, IEEE JSAC, 2022.
6. M. Richards — *"Fundamentals of Radar Signal Processing"*, McGraw-Hill, 2014. [SAR theory]

---

*Δημοκρίτειο Πανεπιστήμιο Θράκης · Αρχές Τηλεπικοινωνιακών Συστημάτων*
