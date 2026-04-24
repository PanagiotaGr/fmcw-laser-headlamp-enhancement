import matplotlib
matplotlib.use("Agg")
# -*- coding: utf-8 -*-
"""
ΙΔΕΕΣ 6, 11, 12, 17 — Υλοποιήσεις
====================================
6:  Danger-adaptive scan mode
11: Pedestrian trajectory prediction
12: Crossing intention prediction
17: Self-healing phase codes
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

np.random.seed(42)

c  = 3e8
fc = 193.4e12
B0 = 10e9
T0 = 10e-6

# ═══════════════════════════════════════════════════════════════════════════
# ΙΔΕΑ 6 — Danger-adaptive scan mode
# ═══════════════════════════════════════════════════════════════════════════

def crlb_range(B, T, M, snr_lin):
    Tc = M * T
    return (c / (2*B)) * np.sqrt(3 / (8 * np.pi**2 * snr_lin * M * Tc**2))

def ber_dpsk(snr_lin):
    return 0.5 * np.exp(-snr_lin)

scan_modes = {
    "LOW danger":  {"B": 5e9,  "T": 20e-6, "M": 20,  "Rb": 0.5e9, "color": "#1D9E75"},
    "MED danger":  {"B": 10e9, "T": 10e-6, "M": 50,  "Rb": 1.0e9, "color": "#BA7517"},
    "HIGH danger": {"B": 20e9, "T": 5e-6,  "M": 100, "Rb": 1.5e9, "color": "#E24B4A"},
}

snr_lin = 10**(10/10)
snr_db_range = np.linspace(0, 20, 60)
snr_lin_range = 10**(snr_db_range/10)

print("=" * 64)
print("ΙΔΕΑ 6 — Danger-Adaptive Scan Mode")
print("=" * 64)
print(f"{'Mode':<18} {'B(GHz)':<10} {'M':<6} {'σ_R(cm)':<12} {'Power(dBm)':<12}")
print("-" * 60)
for mode, p in scan_modes.items():
    sr = crlb_range(p["B"], p["T"], p["M"], snr_lin) * 100
    power_norm = (p["B"]/B0) * (T0/p["T"])  # relative power proxy
    print(f"{mode:<18} {p['B']/1e9:<10.0f} {p['M']:<6} {sr:<12.3f} {10*np.log10(power_norm):<12.2f}")

# ═══════════════════════════════════════════════════════════════════════════
# ΙΔΕΑ 11 — Pedestrian Trajectory Prediction (Social Force Model)
# ═══════════════════════════════════════════════════════════════════════════

class PedestrianSFM:
    """
    Social Force Model (Helbing & Molnar, 1995) για πρόβλεψη τροχιάς πεζού.

    F_total = F_desired + F_social + F_obstacle

    F_desired: πεζός κινείται προς destination με desired speed v0
    F_social:  αποφυγή άλλων πεζών (repulsive Gaussian)
    F_obstacle: αποφυγή εμποδίων
    """
    def __init__(self, pos, vel, dest, v0=1.3, tau=0.5, mass=70):
        self.pos  = np.array(pos,  dtype=float)
        self.vel  = np.array(vel,  dtype=float)
        self.dest = np.array(dest, dtype=float)
        self.v0   = v0
        self.tau  = tau
        self.mass = mass

    def desired_force(self):
        d = self.dest - self.pos
        dn = d / (np.linalg.norm(d) + 1e-6)
        return self.mass * (self.v0 * dn - self.vel) / self.tau

    def social_force(self, others, A=2000, B=0.08):
        F = np.zeros(2)
        for other in others:
            diff = self.pos - other.pos
            dist = np.linalg.norm(diff)
            if dist < 0.1: continue
            F += A * np.exp(-dist/B) * diff / dist
        return F

    def step(self, dt, others=None):
        F = self.desired_force()
        if others:
            F += self.social_force(others)
        acc = F / self.mass
        self.vel += acc * dt
        speed = np.linalg.norm(self.vel)
        if speed > 2*self.v0:
            self.vel = self.vel / speed * 2*self.v0
        self.pos += self.vel * dt

def predict_trajectory(ped, dt=0.1, horizon=2.0, others=None):
    """Προβλέπει τροχιά για horizon δευτερόλεπτα."""
    import copy
    ped_copy = copy.deepcopy(ped)
    steps = int(horizon / dt)
    traj  = [ped_copy.pos.copy()]
    for _ in range(steps):
        ped_copy.step(dt, others)
        traj.append(ped_copy.pos.copy())
    return np.array(traj)

# Σενάριο: πεζός πλησιάζει στη διαδρομή αυτοκινήτου
dt_sim = 0.05
t_total = 3.0
steps_total = int(t_total / dt_sim)

ped = PedestrianSFM(pos=[-5, 4], vel=[1.2, -0.5], dest=[10, 0], v0=1.3)
true_traj = [ped.pos.copy()]

for _ in range(steps_total):
    ped.step(dt_sim)
    true_traj.append(ped.pos.copy())
true_traj = np.array(true_traj)

# Προβλέψεις από διάφορα σημεία στη διαδρομή
pred_horizons = [0.5, 1.0, 2.0]  # seconds ahead
pred_start_idx = int(0.5 / dt_sim)

ped_at_pred = PedestrianSFM(
    pos=true_traj[pred_start_idx],
    vel=(true_traj[pred_start_idx] - true_traj[pred_start_idx-1]) / dt_sim,
    dest=[10, 0]
)
predictions = {h: predict_trajectory(ped_at_pred, dt_sim, h) for h in pred_horizons}

# ADE (Average Displacement Error)
ade = {}
for h in pred_horizons:
    steps_h = int(h / dt_sim)
    end_idx = min(pred_start_idx + steps_h, len(true_traj)-1)
    pred_end = predictions[h][:end_idx - pred_start_idx + 1]
    true_part = true_traj[pred_start_idx: pred_start_idx + len(pred_end)]
    min_len = min(len(pred_end), len(true_part))
    ade[h] = np.mean(np.linalg.norm(pred_end[:min_len] - true_part[:min_len], axis=1))

print("\n" + "=" * 64)
print("ΙΔΕΑ 11 — Pedestrian Trajectory Prediction (SFM)")
print("=" * 64)
for h, err in ade.items():
    print(f"Prediction horizon {h:.1f}s: ADE = {err:.3f} m")

# ═══════════════════════════════════════════════════════════════════════════
# ΙΔΕΑ 12 — Crossing Intention Prediction
# ═══════════════════════════════════════════════════════════════════════════
"""
Features που χρησιμοποιούνται:
- heading_angle_to_road: η γωνία πορείας προς τον δρόμο
- speed: ταχύτητα πεζού
- dist_to_curb: απόσταση από το πεζοδρόμιο
- lateral_velocity: συνιστώσα ταχύτητας προς δρόμο

Classifier: Logistic Regression (αναλυτικό μοντέλο)
"""

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def generate_crossing_dataset(n=400):
    """Συνθετικό dataset πεζών: crossing (1) vs not crossing (0)."""
    X, y = [], []
    for _ in range(n//2):
        # Crossing
        heading = np.random.uniform(150, 210)  # προς δρόμο
        speed   = np.random.uniform(0.8, 1.8)
        dist    = np.random.uniform(0.2, 3.0)
        lat_v   = speed * np.abs(np.sin(np.deg2rad(heading - 180)))
        X.append([heading/360, speed/3, dist/10, lat_v/2])
        y.append(1)
    for _ in range(n//2):
        # Not crossing
        heading = np.random.uniform(0, 100)    # παράλληλα ή αντίθετα
        speed   = np.random.uniform(0.3, 1.5)
        dist    = np.random.uniform(2.0, 8.0)
        lat_v   = speed * np.abs(np.sin(np.deg2rad(heading)))
        X.append([heading/360, speed/3, dist/10, lat_v/2])
        y.append(0)
    idx = np.random.permutation(n)
    return np.array(X)[idx], np.array(y)[idx]

def logistic_train(X, y, lr=0.1, epochs=500):
    m, n = X.shape
    w = np.zeros(n)
    b = 0.0
    for _ in range(epochs):
        z     = X @ w + b
        p     = sigmoid(z)
        grad_w = X.T @ (p - y) / m
        grad_b = np.mean(p - y)
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b

X_data, y_data = generate_crossing_dataset(600)
split = 480
X_tr, y_tr = X_data[:split], y_data[:split]
X_te, y_te = X_data[split:], y_data[split:]

w_lr, b_lr = logistic_train(X_tr, y_tr)
y_pred_prob = sigmoid(X_te @ w_lr + b_lr)
y_pred      = (y_pred_prob > 0.5).astype(int)
accuracy    = np.mean(y_pred == y_te)

# ROC curve
thresholds = np.linspace(0, 1, 200)
tpr_list, fpr_list = [], []
for thr in thresholds:
    yp = (y_pred_prob > thr).astype(int)
    tp = np.sum((yp == 1) & (y_te == 1))
    fp = np.sum((yp == 1) & (y_te == 0))
    fn = np.sum((yp == 0) & (y_te == 1))
    tn = np.sum((yp == 0) & (y_te == 0))
    tpr_list.append(tp / (tp + fn + 1e-9))
    fpr_list.append(fp / (fp + tn + 1e-9))

auc = -np.trapezoid(tpr_list, fpr_list)

print("\n" + "=" * 64)
print("ΙΔΕΑ 12 — Crossing Intention Prediction")
print("=" * 64)
print(f"Classifier: Logistic Regression")
print(f"Features:   heading, speed, dist_to_curb, lateral_velocity")
print(f"Test accuracy: {accuracy*100:.1f}%")
print(f"AUC-ROC:       {auc:.3f}")
print(f"Weights:       {[f'{wi:.3f}' for wi in w_lr]}")

# ═══════════════════════════════════════════════════════════════════════════
# ΙΔΕΑ 17 — Self-healing Phase Codes
# ═══════════════════════════════════════════════════════════════════════════

def rain_attenuation_db(rain_rate_mmh, range_km):
    """
    ITU-R P.838 simplified: α(R) = k × R^α
    Για 1550 nm: k≈0.002, α≈1.0 (approximate)
    """
    k, a_coef = 0.002, 1.0
    return k * (rain_rate_mmh ** a_coef) * range_km

def fog_attenuation_db(visibility_m, range_km):
    """Kim model: β = 3.91/V × (λ/0.55)^(-q)  dB/km"""
    lam_um = 1.55
    if visibility_m > 500:
        q = 1.6
    elif visibility_m > 50:
        q = 1.3
    else:
        q = 0.585 * (visibility_m/1000)**(1/3)
    beta = (3.91 / (visibility_m/1000)) * (lam_um/0.55)**(-q)
    return beta * range_km

modes = {
    "CLEAR (default)": {
        "Rb": 1e9, "B": 10e9, "T": 10e-6, "M": 50,
        "desc": "1 Gbps DPSK, B=10GHz, M=50"
    },
    "RAIN (adapted)": {
        "Rb": 0.5e9, "B": 10e9, "T": 20e-6, "M": 100,
        "desc": "0.5 Gbps, longer T, more chirps"
    },
    "DENSE FOG (adapted)": {
        "Rb": 0.25e9, "B": 5e9, "T": 40e-6, "M": 200,
        "desc": "0.25 Gbps, reduced B, many chirps"
    },
}

range_km = 0.05  # 50 m
weather_scenarios = [
    ("Clear",       0.0,    10000),
    ("Light rain",  5.0,    5000),
    ("Heavy rain",  50.0,   2000),
    ("Dense fog",   0.0,    100),
]

print("\n" + "=" * 64)
print("ΙΔΕΑ 17 — Self-Healing Phase Codes")
print("=" * 64)
base_snr_dB = 12.0  # lower so modes switch

snr_db_results = []
mode_selected  = []
ber_results    = []

for name, rain_r, vis in weather_scenarios:
    att  = rain_attenuation_db(rain_r, range_km) if rain_r > 0 else 0
    att += fog_attenuation_db(vis, range_km) if vis < 5000 else 0
    snr  = base_snr_dB - att

    if snr > 12:
        mode = "CLEAR (default)"
    elif snr > 5:
        mode = "RAIN (adapted)"
    else:
        mode = "DENSE FOG (adapted)"

    snr_lin = 10**(snr/10)
    ber     = ber_dpsk(max(snr_lin, 0.01))

    snr_db_results.append(snr)
    mode_selected.append(mode)
    ber_results.append(ber)

    print(f"{name:<18} Att={att:.2f}dB  SNR={snr:.1f}dB  Mode={mode.split()[0]}  BER={ber:.2e}")

# ─── MEGA FIGURE ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
fig.suptitle("Ιδέες 6, 11, 12, 17 — Adaptive ISCAI Extensions",
             fontsize=13, fontweight="bold")

# ── Ιδέα 6: CRLB per mode ──
ax = axes[0, 0]
for mode, p in scan_modes.items():
    sr_arr = [crlb_range(p["B"], p["T"], p["M"], sl)*100 for sl in snr_lin_range]
    ax.semilogy(snr_db_range, sr_arr, lw=2.5, color=p["color"], label=mode)
ax.axvline(10, color="gray", ls=":", lw=1)
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("σ_R (cm)")
ax.set_title("Ιδέα 6: CRLB ανά Danger Mode")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# ── Ιδέα 6: Bar chart ──
ax = axes[0, 1]
modes6  = list(scan_modes.keys())
sigmas6 = [crlb_range(p["B"], p["T"], p["M"], snr_lin)*100 for p in scan_modes.values()]
colors6 = [p["color"] for p in scan_modes.values()]
bars = ax.bar(modes6, sigmas6, color=colors6, edgecolor="black", linewidth=0.6)
for b, v in zip(bars, sigmas6):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
            f"{v:.3f}cm", ha="center", va="bottom", fontsize=8)
ax.set_ylabel("σ_R (cm)")
ax.set_title("Ιδέα 6: Ακρίβεια ανά Mode")
ax.set_xticklabels(modes6, rotation=10, fontsize=8)
ax.grid(axis="y", alpha=0.3)

# ── Ιδέα 11: Trajectory prediction ──
ax = axes[0, 2]
ax.plot(true_traj[:, 0], true_traj[:, 1], "k-", lw=2, label="Ground truth")
ax.plot(true_traj[pred_start_idx, 0], true_traj[pred_start_idx, 1],
        "ko", ms=8, zorder=5, label="Prediction start")
cols_pred = ["#1D9E75", "#BA7517", "#E24B4A"]
for (h, pred), col in zip(predictions.items(), cols_pred):
    ax.plot(pred[:, 0], pred[:, 1], "--", color=col, lw=2,
            label=f"{h}s ahead (ADE={ade[h]:.2f}m)")
ax.axhline(0, color="gray", ls=":", lw=1, label="Δρόμος")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("Ιδέα 11: Pedestrian Prediction (SFM)")
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

# ── Ιδέα 11: ADE vs horizon ──
ax = axes[0, 3]
hs   = list(ade.keys())
ades = list(ade.values())
ax.bar(hs, ades, width=0.3, color="#185FA5", edgecolor="black", linewidth=0.6)
for xi, yi in zip(hs, ades):
    ax.text(xi, yi + 0.01, f"{yi:.3f}m", ha="center", va="bottom", fontsize=9)
ax.set_xlabel("Prediction horizon (s)")
ax.set_ylabel("ADE (m)")
ax.set_title("Ιδέα 11: ADE vs Horizon")
ax.grid(axis="y", alpha=0.3)

# ── Ιδέα 12: ROC curve ──
ax = axes[1, 0]
ax.plot(fpr_list, tpr_list, lw=2.5, color="#534AB7", label=f"Logistic Reg. (AUC={auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.5)")
ax.fill_between(fpr_list, tpr_list, alpha=0.1, color="#534AB7")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title(f"Ιδέα 12: Crossing Intention ROC (Acc={accuracy*100:.1f}%)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# ── Ιδέα 12: Feature importance ──
ax = axes[1, 1]
feat_names = ["Heading\n(norm)", "Speed\n(norm)", "Dist to\ncurb", "Lateral\nvelocity"]
ax.barh(feat_names, np.abs(w_lr), color=["#185FA5","#1D9E75","#BA7517","#E24B4A"],
        edgecolor="black", linewidth=0.6)
ax.set_xlabel("|Weight|")
ax.set_title("Ιδέα 12: Feature Importance")
ax.grid(axis="x", alpha=0.3)

# ── Ιδέα 17: SNR vs weather ──
ax = axes[1, 2]
w_names = [w[0] for w in weather_scenarios]
colors_w = ["#1D9E75", "#185FA5", "#E24B4A", "#888780"]
bars = ax.bar(w_names, snr_db_results, color=colors_w, edgecolor="black", linewidth=0.6)
for b, v, m in zip(bars, snr_db_results, mode_selected):
    ax.text(b.get_x() + b.get_width()/2, max(v, 0) + 0.3,
            m.split()[0], ha="center", va="bottom", fontsize=8, rotation=0)
ax.axhline(12, color="orange", ls="--", lw=1.5, label="Mode switch @ 12dB")
ax.axhline(5, color="red", ls="--", lw=1.5, label="Mode switch @ 5dB")
ax.set_ylabel("Effective SNR (dB)")
ax.set_title("Ιδέα 17: SNR per Weather — Auto Mode Switch")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

# ── Ιδέα 17: BER vs weather ──
ax = axes[1, 3]
ax.bar(w_names, ber_results, color=colors_w, edgecolor="black", linewidth=0.6)
for i, (b, v) in enumerate(zip(ax.patches, ber_results)):
    ax.text(b.get_x() + b.get_width()/2, v*1.5,
            f"{v:.1e}", ha="center", va="bottom", fontsize=8)
ax.set_yscale("log")
ax.set_ylabel("BER")
ax.set_title("Ιδέα 17: BER per Weather (adaptive mode)")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("/home/claude/iscai_extensions/ideas_06_11_12_17.png", dpi=100, bbox_inches="tight")
plt.close()
print("\nSaved: ideas_06_11_12_17.png")
