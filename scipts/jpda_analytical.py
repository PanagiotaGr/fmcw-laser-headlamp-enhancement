"""
===============================================================================
ΙΔΕΑ 10 — JPDA (Joint Probabilistic Data Association)
Αναλυτική υλοποίηση με μαθηματικό υπόβαθρο
===============================================================================

Θεωρία
------

Στόχος: Αντιστοίχιση measurements z_i σε target state x.

Baseline (Hard Association):
    x_hat = argmin ||z_i - x||

JPDA:
    Υπολογίζουμε πιθανότητες:

    p(z_i | x) ∝ exp( - ||z_i - x||^2 / (2σ^2) )

    Κανονικοποίηση:
    w_i = p(z_i | x) / sum_j p(z_j | x)

    Εκτίμηση:
    x_hat = Σ w_i * z_i

Ιδιότητες:
- πιο robust σε noise
- καλύτερο σε multi-target scenarios
===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. Likelihood function
# -----------------------------------------------------------------------------
def likelihood(z, x, sigma):
    """
    p(z|x) = exp( -||z-x||^2 / (2σ^2) )
    """
    d = np.linalg.norm(z - x)
    return np.exp(-0.5 * (d / sigma)**2)

# -----------------------------------------------------------------------------
# 2. JPDA update
# -----------------------------------------------------------------------------
def jpda_update(measurements, x_pred, sigma):
    weights = np.array([likelihood(z, x_pred, sigma) for z in measurements])
    weights = weights / np.sum(weights)

    x_hat = np.sum(weights[:, None] * measurements, axis=0)
    return x_hat, weights

# -----------------------------------------------------------------------------
# 3. Hard association
# -----------------------------------------------------------------------------
def hard_update(measurements, x_pred):
    dists = np.linalg.norm(measurements - x_pred, axis=1)
    return measurements[np.argmin(dists)]

# -----------------------------------------------------------------------------
# 4. Simulation setup
# -----------------------------------------------------------------------------
np.random.seed(0)

true_target = np.array([50.0, 10.0])
sigma_noise = 3.0

# true measurements
meas = true_target + sigma_noise * np.random.randn(6, 2)

# clutter
clutter = np.random.uniform([40, 0], [60, 20], size=(6, 2))

measurements = np.vstack([meas, clutter])

# prediction (με error)
x_pred = true_target + np.array([3.0, -2.0])

# -----------------------------------------------------------------------------
# 5. Apply methods
# -----------------------------------------------------------------------------
x_hard = hard_update(measurements, x_pred)
x_jpda, weights = jpda_update(measurements, x_pred, sigma_noise)

# -----------------------------------------------------------------------------
# 6. Errors
# -----------------------------------------------------------------------------
err_hard = np.linalg.norm(x_hard - true_target)
err_jpda = np.linalg.norm(x_jpda - true_target)

print("="*60)
print("JPDA vs Hard Association")
print("="*60)
print(f"Hard error = {err_hard:.3f}")
print(f"JPDA error = {err_jpda:.3f}")

# -----------------------------------------------------------------------------
# 7. Plot
# -----------------------------------------------------------------------------
plt.figure(figsize=(6,6))

plt.scatter(measurements[:,0], measurements[:,1], alpha=0.6, label="Measurements")
plt.scatter(*true_target, marker='x', s=100, label="True")
plt.scatter(*x_pred, label="Prediction")
plt.scatter(*x_hard, marker='s', label="Hard")
plt.scatter(*x_jpda, marker='^', label="JPDA")

plt.title("JPDA vs Hard Association")
plt.grid()
plt.legend()

plt.show()
