"""
===============================================================================
ΙΔΕΑ 16 — Cognitive ISAC Resource Allocation
Αναλυτική / ημι-αναλυτική υλοποίηση με θεωρία μέσα στο Python αρχείο
===============================================================================

Σκοπός
------
Η παρούσα πρόταση βελτίωσης αφορά τη δυναμική κατανομή πόρων μεταξύ:
  1) sensing
  2) communication

στο baseline PC-FMCW ISCAI σύστημα.

Βασική ιδέα
-----------
Στο baseline paper, το σύστημα λειτουργεί ως integrated sensing and
communication (ISAC), δηλαδή το ίδιο waveform και το ίδιο hardware
υποστηρίζουν και τις δύο λειτουργίες.

Ωστόσο, στην πράξη οι απαιτήσεις δεν είναι σταθερές:
- σε πυκνή κυκλοφορία / πολύπλοκο περιβάλλον απαιτείται ισχυρότερο sensing
- σε αραιή κυκλοφορία μπορεί να δοθεί μεγαλύτερη έμφαση στην επικοινωνία

Για τον λόγο αυτό εισάγεται η παράμετρος:
    α ∈ [0, 1]

όπου:
- α      = ποσοστό πόρων / ισχύος / χρόνου που διατίθεται για sensing
- 1 - α  = ποσοστό πόρων / ισχύος / χρόνου που διατίθεται για communication

Θεωρητικό μοντέλο
-----------------
Χρησιμοποιούμε ένα απλό analytical trade-off model.

1) Sensing metric
-----------------
Υποθέτουμε ότι το sensing SINR αυξάνει ανάλογα με το ποσοστό α:

    SINR_s ∝ α * γ * M

όπου:
- γ = baseline SNR
- M = effective coherent resource factor (π.χ. αριθμός chirps / integration gain)

Στο script χρησιμοποιούμε:

    SINR_s(dB) = 10 log10( α * γ * M )

2) Communication metric
-----------------------
Για το communication part, χρησιμοποιούμε Shannon-style proxy:

    R_c = log2( 1 + (1 - α) * γ )

όπου:
- R_c είναι ο κανονικοποιημένος communication rate (bits/s/Hz)
- (1 - α) * γ είναι το effective communication SNR budget

Ερμηνεία
--------
- Όσο αυξάνει το α, βελτιώνεται το sensing SINR
- Όσο μειώνεται το α, αυξάνεται ο communication rate
- Άρα υπάρχει trade-off μεταξύ sensing και communication

Η καμπύλη:
    (R_c(α), SINR_s(α))
μπορεί να ιδωθεί ως Pareto-like σύνολο λειτουργίας.

Προτεινόμενη adaptation policy
------------------------------
Ως απλό policy model, θεωρούμε ότι το βέλτιστο α εξαρτάται από την
πυκνότητα κυκλοφορίας traffic_density ∈ [0, 1]:

    α_opt = 0.3 + 0.5 * traffic_density

Άρα:
- low traffic  -> μικρότερο α -> περισσότερο communication
- high traffic -> μεγαλύτερο α -> περισσότερο sensing

Κατηγορία εγκυρότητας
---------------------
[A/S] Analytical / Semi-analytical

- Το trade-off model είναι analytical.
- Το policy α_opt είναι simple engineering heuristic.
===============================================================================
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. Baseline παράμετροι
# -----------------------------------------------------------------------------
# gamma_lin : baseline SNR σε γραμμική κλίμακα
# M_eff     : effective coherent sensing resource factor
#             (abstract gain parameter για να φαίνεται η ενίσχυση του sensing)

gamma_lin = 10.0
M_eff = 100.0

# -----------------------------------------------------------------------------
# 2. Θεωρητικές συναρτήσεις
# -----------------------------------------------------------------------------
def sensing_sinr_db(alpha: np.ndarray, gamma: float = gamma_lin,
                    M: float = M_eff) -> np.ndarray:
    """
    Sensing analytical metric

    Υποθέτουμε:
        SINR_s = α * γ * M

    Σε dB:
        SINR_s(dB) = 10 log10( α * γ * M )
    """
    alpha = np.asarray(alpha)
    return 10.0 * np.log10(np.maximum(alpha * gamma * M, 1e-12))


def communication_rate(alpha: np.ndarray, gamma: float = gamma_lin) -> np.ndarray:
    """
    Communication analytical proxy

    Υποθέτουμε:
        R_c = log2(1 + (1 - α) * γ)

    όπου το (1-α) * γ παίζει ρόλο effective communication SNR budget.
    """
    alpha = np.asarray(alpha)
    return np.log2(1.0 + np.maximum((1.0 - alpha) * gamma, 1e-12))


def alpha_opt_from_traffic(traffic_density: float) -> float:
    """
    Απλό heuristic policy για adaptive ISAC resource allocation.

    α_opt = 0.3 + 0.5 * traffic_density
    με saturation στο [0, 1]
    """
    return float(np.clip(0.3 + 0.5 * traffic_density, 0.0, 1.0))


# -----------------------------------------------------------------------------
# 3. Sweep της παραμέτρου α
# -----------------------------------------------------------------------------
alpha_vals = np.linspace(0.01, 0.99, 300)

sinr_vals = sensing_sinr_db(alpha_vals)
rate_vals = communication_rate(alpha_vals)

# -----------------------------------------------------------------------------
# 4. Δύο ενδεικτικά σενάρια κυκλοφορίας
# -----------------------------------------------------------------------------
traffic_low = 0.2
traffic_high = 0.8

alpha_low = alpha_opt_from_traffic(traffic_low)
alpha_high = alpha_opt_from_traffic(traffic_high)

sinr_low = sensing_sinr_db(alpha_low)
sinr_high = sensing_sinr_db(alpha_high)

rate_low = communication_rate(alpha_low)
rate_high = communication_rate(alpha_high)

# -----------------------------------------------------------------------------
# 5. Baseline σταθερό operating point
# -----------------------------------------------------------------------------
# Για λόγους σύγκρισης, θεωρούμε ένα fixed split:
#   α_base = 0.5
#
# Αυτό είναι απλό σημείο αναφοράς. Δεν σημαίνει ότι το paper δίνει ακριβώς
# αυτόν τον τύπο, αλλά μας βοηθά να συγκρίνουμε fixed vs adaptive allocation.

alpha_base = 0.5
sinr_base = sensing_sinr_db(alpha_base)
rate_base = communication_rate(alpha_base)

# -----------------------------------------------------------------------------
# 6. Normalized utility function (προαιρετική)
# -----------------------------------------------------------------------------
# Για να φανεί η ιδέα της "βέλτιστης" επιλογής, ορίζουμε ένα weighted utility:
#
#   U(α) = w_s * normalized(SINR_s) + w_c * normalized(R_c)
#
# όπου:
# - w_s = βάρος sensing
# - w_c = βάρος communication
#
# Τα βάρη εξαρτώνται από την πυκνότητα κυκλοφορίας:
#   w_s = 0.3 + 0.5 * traffic_density
#   w_c = 1 - w_s
#
# Αυτό είναι engineering heuristic, όχι θεμελιώδης νόμος.

def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-12)

def utility(alpha: np.ndarray, traffic_density: float,
            gamma: float = gamma_lin, M: float = M_eff) -> np.ndarray:
    s = normalize(sensing_sinr_db(alpha, gamma, M))
    r = normalize(communication_rate(alpha, gamma))
    w_s = 0.3 + 0.5 * traffic_density
    w_c = 1.0 - w_s
    return w_s * s + w_c * r

utility_low = utility(alpha_vals, traffic_low)
utility_high = utility(alpha_vals, traffic_high)

alpha_best_low = alpha_vals[np.argmax(utility_low)]
alpha_best_high = alpha_vals[np.argmax(utility_high)]

# -----------------------------------------------------------------------------
# 7. Εκτύπωση αποτελεσμάτων
# -----------------------------------------------------------------------------
print("=" * 78)
print("ΙΔΕΑ 16 — Cognitive ISAC Resource Allocation")
print("=" * 78)
print(f"Baseline SNR γ = {gamma_lin:.2f}")
print(f"Effective sensing factor M = {M_eff:.1f}")
print()
print("Fixed baseline operating point:")
print(f"  α_base = {alpha_base:.2f}")
print(f"  Sensing SINR = {sinr_base:.3f} dB")
print(f"  Communication rate = {rate_base:.3f} bits/s/Hz")
print()
print("Adaptive scenarios:")
print(f"  Low traffic  (density={traffic_low:.1f})  -> α_opt = {alpha_low:.3f}")
print(f"     Sensing SINR = {sinr_low:.3f} dB")
print(f"     Comm rate    = {rate_low:.3f} bits/s/Hz")
print()
print(f"  High traffic (density={traffic_high:.1f}) -> α_opt = {alpha_high:.3f}")
print(f"     Sensing SINR = {sinr_high:.3f} dB")
print(f"     Comm rate    = {rate_high:.3f} bits/s/Hz")
print()
print("Utility-based α selection (heuristic):")
print(f"  Best α for low traffic  = {alpha_best_low:.3f}")
print(f"  Best α for high traffic = {alpha_best_high:.3f}")

# -----------------------------------------------------------------------------
# 8. Διαγράμματα
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
fig.suptitle(
    "Ιδέα 16 — Cognitive ISAC Resource Allocation",
    fontsize=12,
    fontweight="bold"
)

# Plot 1: Sensing και Communication vs alpha
ax = axes[0]
ax.plot(alpha_vals, normalize(sinr_vals), linewidth=2, label="Normalized sensing SINR")
ax.plot(alpha_vals, normalize(rate_vals), linewidth=2, linestyle="--", label="Normalized communication rate")
ax.axvline(alpha_base, color="black", linestyle=":", linewidth=1.2, label="Fixed baseline")
ax.axvline(alpha_low, color="green", linestyle="--", linewidth=1.2, label="Low traffic α")
ax.axvline(alpha_high, color="red", linestyle="--", linewidth=1.2, label="High traffic α")
ax.set_xlabel("α (resource fraction for sensing)")
ax.set_ylabel("Normalized metric")
ax.set_title("Μεταβολή sensing / communication με το α")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

# Plot 2: Trade-off curve
ax = axes[1]
ax.plot(rate_vals, sinr_vals, linewidth=2, label="Trade-off curve")
ax.scatter(rate_base, sinr_base, s=70, label="Fixed baseline")
ax.scatter(rate_low, sinr_low, s=70, label="Low traffic")
ax.scatter(rate_high, sinr_high, s=70, label="High traffic")
ax.set_xlabel("Communication rate (bits/s/Hz)")
ax.set_ylabel("Sensing SINR (dB)")
ax.set_title("ISAC trade-off")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

# Plot 3: Utility functions
ax = axes[2]
ax.plot(alpha_vals, utility_low, linewidth=2, label="Utility (low traffic)")
ax.plot(alpha_vals, utility_high, linewidth=2, linestyle="--", label="Utility (high traffic)")
ax.axvline(alpha_best_low, color="green", linestyle=":", linewidth=1.2)
ax.axvline(alpha_best_high, color="red", linestyle=":", linewidth=1.2)
ax.set_xlabel("α (resource fraction for sensing)")
ax.set_ylabel("Utility")
ax.set_title("Heuristic utility-based επιλογή του α")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

fig.tight_layout(rect=[0, 0, 1, 0.92])

# -----------------------------------------------------------------------------
# 9. Αποθήκευση
# -----------------------------------------------------------------------------
out_dir = Path("/mnt/data")
fig_path = out_dir / "idea16_cognitive_isac.png"
py_path = out_dir / "idea16_cognitive_isac.py"
txt_path = out_dir / "idea16_cognitive_isac_paragraph.txt"

fig.savefig(fig_path, dpi=170, bbox_inches="tight")
plt.close(fig)

paragraph = """
Ιδέα 16 — Cognitive ISAC Resource Allocation

Ως βελτίωση του baseline ISAC συστήματος προτείνεται η δυναμική κατανομή
πόρων μεταξύ sensing και communication μέσω μιας παραμέτρου α, όπου το α
εκφράζει το ποσοστό των πόρων που διατίθενται για sensing και το 1−α το
αντίστοιχο ποσοστό για communication. Με analytical μοντέλο θεωρείται ότι
το sensing SINR ακολουθεί τη σχέση SINR_s ∝ α·γ·M, ενώ ο communication
ρυθμός προσεγγίζεται από τη σχέση R_c = log2(1 + (1−α)γ). Η ανάλυση δείχνει
ότι καθώς αυξάνεται το α, βελτιώνεται η sensing απόδοση εις βάρος του
communication rate, γεγονός που οδηγεί σε σαφή trade-off μεταξύ των δύο
λειτουργιών. Επιπλέον, η επιλογή του βέλτιστου α μπορεί να γίνει δυναμικά
με βάση τις συνθήκες κυκλοφορίας, ώστε σε περιβάλλοντα υψηλής πολυπλοκότητας
να ενισχύεται το sensing και σε απλούστερα περιβάλλοντα να ενισχύεται η
επικοινωνία. Η προσέγγιση αυτή συνιστά μια φυσική system-level επέκταση του
baseline συστήματος προς πιο ευφυή και προσαρμοστική λειτουργία.
""".strip() + "\n"

py_path.write_text(__doc__ + "\n", encoding="utf-8")
txt_path.write_text(paragraph, encoding="utf-8")

print()
print(f"Saved figure: {fig_path}")
print(f"Saved paragraph: {txt_path}")
