import matplotlib
matplotlib.use("Agg")
# -*- coding: utf-8 -*-
"""
ΙΔΕΑ 1 — Δύο Αμάξια ως Virtual Giant Sensor (Synthetic Aperture)
=================================================================
Δύο αυτοκίνητα με FMCW laser headlamps ενώνουν τα σήματά τους
σαν ένα μεγαλύτερο aperture (Synthetic Aperture Radar / SAR).

Θεωρία
------
Ένα μόνο αμάξι έχει aperture D → angular resolution θ = λ/D
Δύο αμάξια με baseline B → virtual aperture = B → θ_virtual = λ/B

Αν B >> D, η angular resolution βελτιώνεται κατά B/D φορές.

Για range resolution (FMCW): ΔR = c/(2B_chirp)  (αμετάβλητο)
Για cross-range resolution:  Δx = R·λ/L_aperture

Πρακτικό μοντέλο
----------------
Αμάξι Α στο x=0, Αμάξι Β στο x=d_baseline
Κάθε αμάξι μετράει ένα beat signal από τον ίδιο στόχο.
Τα signals συνδυάζονται (coherent combination) για να παράγουν
ένα "virtual" σήμα με διπλή (ή N-πλή) ανάλυση cross-range.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

np.random.seed(42)

# ─── Παράμετροι συστήματος ────────────────────────────────────────────────
c       = 3e8          # m/s
fc      = 193.4e12     # Hz  (1551 nm)
lam     = c / fc       # ~1.55e-6 m
B       = 10e9         # Hz  chirp bandwidth
T       = 10e-6        # s   chirp period
mu      = B / T        # Hz/s
N       = 256          # samples per chirp
M       = 64           # chirps per CPI
fs      = 2 * B
dt      = 1 / fs
SNR_dB  = 15
sigma   = np.sqrt(1 / (2 * 10**(SNR_dB/10)))

# ─── Γεωμετρία ────────────────────────────────────────────────────────────
D_aperture   = 0.05    # m  (5 cm, single car aperture)
d_baseline   = 2.0     # m  (απόσταση μεταξύ αμαξιών)
# Virtual aperture ≈ d_baseline

# ─── Στόχοι ───────────────────────────────────────────────────────────────
# Δύο στόχοι κοντά σε γωνία — το single-car δεν τους ξεχωρίζει
targets = [
    {"R": 40.0, "x_cross": -0.3, "v": 0.0},   # στόχος αριστερά
    {"R": 40.0, "x_cross": +0.3, "v": 0.0},   # στόχος δεξιά
    # (ίδια απόσταση, 0.6m cross-range separation)
]

def angular_resolution(aperture_m, range_m):
    """Cross-range resolution Δx = R·λ / L_aperture"""
    return range_m * lam / aperture_m

def build_beat_signal(car_x, targets, M_chirps, N_samp):
    """
    Παράγει M×N beat signal matrix για ένα αμάξι στη θέση car_x.
    Κάθε στόχος συνεισφέρει με phase offset ανάλογο της γεωμετρίας.
    """
    S = np.zeros((N_samp, M_chirps), dtype=complex)
    t = np.arange(N_samp) * dt

    for tgt in targets:
        R  = tgt["R"]
        xc = tgt["x_cross"]
        v  = tgt["v"]

        # Slant range από car_x
        R_slant = np.sqrt(R**2 + (xc - car_x)**2)
        tau     = 2 * R_slant / c
        f_beat  = mu * tau
        f_D     = 2 * v / lam

        for m in range(M_chirps):
            phase_slow = 2 * np.pi * f_D * m * T
            s = np.exp(1j * (2*np.pi*f_beat*t + phase_slow))
            S[:, m] += s

    # AWGN
    S += sigma * (np.random.randn(*S.shape) + 1j*np.random.randn(*S.shape)) / np.sqrt(2)
    return S

# ─── Σήματα από τα δύο αμάξια ─────────────────────────────────────────────
S_carA = build_beat_signal(car_x=0.0,          targets=targets, M_chirps=M, N_samp=N)
S_carB = build_beat_signal(car_x=d_baseline,   targets=targets, M_chirps=M, N_samp=N)

# ─── Range FFT ────────────────────────────────────────────────────────────
def range_fft(S):
    return np.fft.fft(S, axis=0)

RA = range_fft(S_carA)
RB = range_fft(S_carB)

# ─── Coherent combination (virtual aperture) ──────────────────────────────
# Για coherent SAR: phase-align και αθροίζω
# Απλοποιημένο: conjugate multiply (cross-correlation στον aperture axis)
R_virtual = RA + RB   # coherent sum → signal αυξάνει 2×, noise √2× → SNR +3dB
R_single  = RA        # μόνο ένα αμάξι

# ─── Cross-range (Doppler / angle) FFT ────────────────────────────────────
def doppler_fft(R):
    return np.fft.fftshift(np.fft.fft(R, axis=1), axes=1)

RDmap_single  = np.abs(doppler_fft(R_single))**2
RDmap_virtual = np.abs(doppler_fft(R_virtual))**2

# ─── Μετατροπή σε range / cross-range axes ────────────────────────────────
freq_bins   = np.fft.fftfreq(N, dt)[:N//2]
range_axis  = freq_bins * c / (2 * mu)   # meters

doppler_bins   = np.fft.fftshift(np.fft.fftfreq(M, T))
target_R       = targets[0]["R"]
cross_range    = target_R * lam * doppler_bins / (2 * d_baseline)  # approx

# ─── Υπολογισμός ανάλυσης ─────────────────────────────────────────────────
dx_single  = angular_resolution(D_aperture, target_R)
dx_virtual = angular_resolution(d_baseline, target_R)
improvement = dx_single / dx_virtual

print("=" * 64)
print("ΙΔΕΑ 1 — Virtual Giant Sensor (Synthetic Aperture)")
print("=" * 64)
print(f"Baseline μεταξύ αμαξιών:       {d_baseline:.1f} m")
print(f"Single-car aperture:            {D_aperture*100:.0f} cm")
print(f"Virtual aperture:               {d_baseline:.1f} m")
print(f"Target range:                   {target_R:.0f} m")
print(f"Cross-range resolution (single):{dx_single:.3f} m")
print(f"Cross-range resolution (virtual):{dx_virtual:.4f} m")
print(f"Βελτίωση ανάλυσης:              {improvement:.0f}×")
print(f"SNR κέρδος (coherent sum):      +{10*np.log10(2):.1f} dB")

# ─── Figure ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Ιδέα 1 — Δύο Αμάξια ως Virtual Giant Sensor\n"
             f"Baseline = {d_baseline} m  |  Βελτίωση ανάλυσης = {improvement:.0f}×",
             fontsize=12, fontweight="bold")

R_idx = np.argmin(np.abs(range_axis - target_R))

# Plot 1: Range-Doppler map single car
ax = axes[0]
ax.imshow(10*np.log10(RDmap_single[:N//2] + 1e-9), aspect="auto",
          extent=[doppler_bins[0], doppler_bins[-1], range_axis[N//4], range_axis[0]],
          cmap="viridis", vmin=-20, vmax=40)
ax.set_xlabel("Doppler (Hz)")
ax.set_ylabel("Range (m)")
ax.set_title("Single car — Range-Doppler")
ax.set_ylim(30, 55)

# Plot 2: Range-Doppler map virtual
ax = axes[1]
ax.imshow(10*np.log10(RDmap_virtual[:N//2] + 1e-9), aspect="auto",
          extent=[doppler_bins[0], doppler_bins[-1], range_axis[N//4], range_axis[0]],
          cmap="viridis", vmin=-20, vmax=40)
ax.set_xlabel("Doppler (Hz)")
ax.set_ylabel("Range (m)")
ax.set_title("Virtual sensor (2 cars) — Range-Doppler")
ax.set_ylim(30, 55)

# Plot 3: Cross-range profile at target range
ax = axes[2]
prof_single  = RDmap_single[R_idx,  :]
prof_virtual = RDmap_virtual[R_idx, :]
norm_s = prof_single  / prof_single.max()
norm_v = prof_virtual / prof_virtual.max()
ax.plot(doppler_bins, 10*np.log10(norm_s  + 1e-9), lw=2, label=f"Single car (Δx={dx_single:.2f} m)")
ax.plot(doppler_bins, 10*np.log10(norm_v  + 1e-9), lw=2, ls="--", label=f"Virtual sensor (Δx={dx_virtual:.4f} m)")
ax.axhline(-3, color="gray", ls=":", lw=1, label="−3 dB")
ax.set_xlabel("Doppler bin")
ax.set_ylabel("Normalized power (dB)")
ax.set_title("Cross-range profile @ target range")
ax.set_ylim(-30, 3)
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("/home/claude/iscai_extensions/idea01_virtual_giant_sensor.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: idea01_virtual_giant_sensor.png")
