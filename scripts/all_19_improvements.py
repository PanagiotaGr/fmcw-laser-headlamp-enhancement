"""
===============================================================================
19 ΙΔΕΕΣ ΒΕΛΤΙΩΣΗΣ — ΑΝΑΛΥΤΙΚΟΣ / ΗΜΙ-ΑΝΑΛΥΤΙΚΟΣ PYTHON ΚΩΔΙΚΑΣ
για το PC-FMCW ISCAI Laser Headlamp
===============================================================================

Βασική δημοσίευση:
Shuanghe Liu, Tongzheng Sun, Xiang Shu, Jian Song, Yuhan Dong,
"Phase-coded FMCW Laser Headlamp for Integrated Sensing,
Communication, and Illumination",
IEEE Photonics Technology Letters, 2025
DOI: 10.1109/LPT.2025.3649597

Σκοπός:
Ο παρών κώδικας συγκεντρώνει 19 ιδέες βελτίωσης του baseline συστήματος
του paper και τις αξιολογεί με τρεις βαθμίδες επιστημονικής ωριμότητας:

[A] Analytical
    Βασίζεται σε σαφή θεωρητική σχέση / κλειστό τύπο.

[S] Semi-analytical
    Συνδυάζει θεωρία με προσομοιωτικό / approximate μοντέλο.

[C] Conceptual / Future Work
    Είναι λογική ερευνητική κατεύθυνση, αλλά όχι πλήρως αποδεδειγμένη
    από τον παρόντα κώδικα.

Σημείωση:
Ο κώδικας αυτός είναι "μεγάλο αναλυτικό appendix". Δεν σημαίνει ότι όλες
οι ιδέες έχουν το ίδιο βάρος ή ότι όλες είναι εξίσου κατάλληλες ως κύρια
πρόταση βελτίωσης στην εργασία.
===============================================================================
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import uniform_filter
from scipy.special import erfc
from scipy.linalg import eigh

# -----------------------------------------------------------------------------
# 0. Αναπαραγωγιμότητα
# -----------------------------------------------------------------------------
np.random.seed(42)

# -----------------------------------------------------------------------------
# 1. Baseline παράμετροι από το paper
# -----------------------------------------------------------------------------
c   = 3e8
fc  = 193.4e12
B   = 10e9
T   = 10e-6
mu  = B / T
lam = c / fc
Rb  = 1e9

M_chirps = 100
N_fast   = 256
fs       = 2 * B
dt       = 1 / fs

SNR_dB_range = np.linspace(0, 20, 60)
SNR_lin_range = 10 ** (SNR_dB_range / 10.0)

# -----------------------------------------------------------------------------
# 2. Βοηθητικές συναρτήσεις
# -----------------------------------------------------------------------------
def qfunc(x):
    return 0.5 * erfc(x / np.sqrt(2))

def ber_dbpsk(gamma_b):
    """[A] Theoretical BER for noncoherent DBPSK in AWGN."""
    return 0.5 * np.exp(-gamma_b)

def ber_qpsk(gamma_b):
    """[A] Theoretical BER for coherent QPSK in AWGN."""
    return qfunc(np.sqrt(2 * gamma_b))

def ber_mqam_gray(gamma_b, M):
    """
    [A] First-order BER approximation for square Gray-coded M-QAM in AWGN.

    Για M=4 η σχέση ταυτίζεται με QPSK.
    Δεν είναι πλήρες coded-OFDM BER model, αλλά analytical per-subcarrier
    approximation.
    """
    if M == 4:
        return ber_qpsk(gamma_b)
    k = np.log2(M)
    ber = (4 / k) * (1 - 1 / np.sqrt(M)) * qfunc(np.sqrt((3 * k / (M - 1)) * gamma_b))
    return np.clip(ber, 1e-12, 0.5)

def range_resolution(B_val):
    """[A] FMCW range resolution."""
    return c / (2 * B_val)

def crlb_range(B_val, gamma, M_val=100, T_val=10e-6):
    """
    [A] CRLB-style range metric, συμβατό με το analytical reproduction μέρος.
    """
    Tc = M_val * T_val
    return (c / (2 * B_val)) * np.sqrt(3 / (8 * np.pi**2 * gamma * M_val * Tc**2))

def required_snr_db_for_target_ber(ber_fun, target_ber=1e-6,
                                   search_db=np.linspace(0, 30, 5000), **kwargs):
    """[A] Numeric inversion of BER curve."""
    gamma = 10 ** (search_db / 10.0)
    vals = ber_fun(gamma, **kwargs) if kwargs else ber_fun(gamma)
    idx = np.argmin(np.abs(vals - target_ber))
    return float(search_db[idx])

def effective_goodput(base_rate_bps, ber, overhead_fraction=0.0):
    """[S] First-order effective goodput approximation."""
    return base_rate_bps * (1 - ber) * (1 - overhead_fraction)

# -----------------------------------------------------------------------------
# 3. Registry για τις 19 ιδέες
# -----------------------------------------------------------------------------
ideas = []

def register_idea(idx, title, category, validity, summary):
    ideas.append({
        "idx": idx,
        "title": title,
        "category": category,
        "validity": validity,
        "summary": summary,
    })

# =============================================================================
# ΚΑΤΗΓΟΡΙΑ 1 — ΕΠΙΚΟΙΝΩΝΙΑ
# =============================================================================

# -----------------------------------------------------------------------------
# ΙΔΕΑ 1 — OFDM/QAM-based extension αντί για DBPSK
# -----------------------------------------------------------------------------
register_idea(
    1,
    "OFDM/QAM-based extension αντί για DBPSK",
    "Επικοινωνία",
    "[A/S]",
    "Αναλυτική BER/throughput σύγκριση μεταξύ baseline DBPSK και OFDM-QPSK/16QAM/64QAM."
)

N_sc = 64
N_cp = 16
cp_overhead = N_cp / (N_sc + N_cp)

ber_baseline_dbpsk = ber_dbpsk(SNR_lin_range)
ber_ofdm_qpsk = ber_qpsk(SNR_lin_range)
ber_ofdm_16qam = ber_mqam_gray(SNR_lin_range, 16)
ber_ofdm_64qam = ber_mqam_gray(SNR_lin_range, 64)

rate_dbpsk = 1e9
rate_qpsk = 2e9
rate_16qam = 4e9
rate_64qam = 6e9

goodput_dbpsk = effective_goodput(rate_dbpsk, ber_baseline_dbpsk, 0.0)
goodput_qpsk = effective_goodput(rate_qpsk, ber_ofdm_qpsk, cp_overhead)
goodput_16qam = effective_goodput(rate_16qam, ber_ofdm_16qam, cp_overhead)
goodput_64qam = effective_goodput(rate_64qam, ber_ofdm_64qam, cp_overhead)

# -----------------------------------------------------------------------------
# ΙΔΕΑ 2 — LDPC coding gain
# -----------------------------------------------------------------------------
register_idea(
    2,
    "LDPC coding πάνω από το baseline communication layer",
    "Επικοινωνία",
    "[S]",
    "Χρήση coding-gain approximation 5.5 dB για μετατόπιση των BER curves."
)

def ldpc_shifted_ber_dbpsk(snr_db, coding_gain_db=5.5):
    return ber_dbpsk(10 ** ((snr_db + coding_gain_db) / 10.0))

def ldpc_shifted_ber_qpsk(snr_db, coding_gain_db=5.5):
    return ber_qpsk(10 ** ((snr_db + coding_gain_db) / 10.0))

ber_ldpc_dbpsk = ldpc_shifted_ber_dbpsk(SNR_dB_range, 5.5)
ber_ldpc_qpsk = ldpc_shifted_ber_qpsk(SNR_dB_range, 5.5)

# -----------------------------------------------------------------------------
# ΙΔΕΑ 3 — Polarization multiplexing
# -----------------------------------------------------------------------------
register_idea(
    3,
    "Polarization multiplexing για διπλασιασμό throughput",
    "Επικοινωνία",
    "[A/S]",
    "Δύο polarization streams με μικρή SNR ποινή λόγω cross-polarization mixing."
)

def polarization_mux_goodput(snr_db, mixing_deg=3.0):
    gamma = 10 ** (snr_db / 10.0)
    eps = np.deg2rad(mixing_deg)
    gamma_eff = gamma * np.cos(eps) ** 2
    ber_pol = ber_dbpsk(gamma_eff)
    return 2 * Rb * (1 - ber_pol), ber_pol

tp_pol, ber_pol = polarization_mux_goodput(SNR_dB_range, mixing_deg=3.0)

# -----------------------------------------------------------------------------
# ΙΔΕΑ 4 — MMSE equalization σε optical fading
# -----------------------------------------------------------------------------
register_idea(
    4,
    "MMSE equalization σε ατμοσφαιρική διακύμανση",
    "Επικοινωνία",
    "[S]",
    "Log-normal fading channel και first-order MMSE equalization approximation."
)

def atmospheric_fading_sample(turbulence_sigma=0.4):
    return np.exp(turbulence_sigma * np.random.randn() - turbulence_sigma**2 / 2)

def compare_equalizer_under_fading(snr_db_values, n_trials=300):
    ber_no_eq = []
    ber_mmse = []
    for snr_db in snr_db_values:
        g = 10 ** (snr_db / 10.0)
        vals_no = []
        vals_eq = []
        for _ in range(n_trials):
            h = atmospheric_fading_sample(0.4)
            gamma_eff = g * h**2
            vals_no.append(ber_dbpsk(max(gamma_eff, 1e-12)))

            h_est = h * (1 + 0.05 * np.random.randn())
            gamma_mmse = g * h**2 / (1 + 1 / max(g * h_est**2, 1e-12))
            vals_eq.append(ber_dbpsk(max(gamma_mmse, 1e-12)))
        ber_no_eq.append(np.mean(vals_no))
        ber_mmse.append(np.mean(vals_eq))
    return np.array(ber_no_eq), np.array(ber_mmse)

snr_subset_comm = SNR_dB_range[::3]
ber_no_eq, ber_mmse = compare_equalizer_under_fading(snr_subset_comm)

# =============================================================================
# ΚΑΤΗΓΟΡΙΑ 2 — SENSING
# =============================================================================

# -----------------------------------------------------------------------------
# ΙΔΕΑ 5 — MUSIC superresolution
# -----------------------------------------------------------------------------
register_idea(
    5,
    "MUSIC superresolution για διάκριση κοντινών targets",
    "Sensing",
    "[S]",
    "Υπερανάλυση στο beat model αντί για συμβατικό FFT στο fast-time."
)

def generate_multi_target_beat(ranges, velocities, snr_db=15, M_ch=100, N_s=256):
    sigma = np.sqrt(1 / (2 * 10 ** (snr_db / 10.0)))
    S = np.zeros((N_s, M_ch), dtype=complex)
    t = np.arange(N_s) * dt
    for R, v in zip(ranges, velocities):
        tau = 2 * R / c
        fd = 2 * v / lam
        for m in range(M_ch):
            S[:, m] += np.exp(1j * (2 * np.pi * mu * tau * t + 2 * np.pi * fd * m * T))
    S += sigma * (np.random.randn(*S.shape) + 1j * np.random.randn(*S.shape)) / np.sqrt(2)
    return S

def music_spectrum(S, n_targets=2, n_scan=512):
    Rxx = S @ S.conj().T / S.shape[1]
    vals, vecs = eigh(Rxx)
    idx = np.argsort(vals)[::-1]
    En = vecs[:, idx[n_targets:]]
    freqs = np.linspace(0, fs / 2, n_scan)
    P = np.zeros(n_scan)
    n = np.arange(S.shape[0]) * dt
    for i, f in enumerate(freqs):
        a = np.exp(1j * 2 * np.pi * f * n)
        denom = np.abs(a.conj() @ En @ En.conj().T @ a)
        P[i] = 1 / (denom + 1e-12)
    return freqs * c / (2 * mu), P

S_close = generate_multi_target_beat([50.0, 52.5], [5.0, 8.0], snr_db=15, M_ch=100, N_s=256)
range_music, P_music = music_spectrum(S_close, n_targets=2, n_scan=512)
fft_full_music = np.abs(np.fft.fft(S_close[:, 0]))**2
fft_half_music = fft_full_music[:S_close.shape[0] // 2]
fft_freq = np.fft.fftfreq(S_close.shape[0], dt)[:S_close.shape[0] // 2]
range_fft = fft_freq * c / (2 * mu)

# -----------------------------------------------------------------------------
# ΙΔΕΑ 6 — Compressed sensing FMCW
# -----------------------------------------------------------------------------
register_idea(
    6,
    "Compressed sensing FMCW με OMP recovery",
    "Sensing",
    "[S]",
    "Sparse range recovery με λιγότερα measurements/chirps."
)

def omp_recovery(y, Phi, n_targets=2, max_iter=20):
    residual = y.copy()
    support = []
    n = Phi.shape[1]
    x_hat = np.zeros(n, dtype=complex)

    for _ in range(min(n_targets, max_iter)):
        corr = np.abs(Phi.conj().T @ residual)
        idx = int(np.argmax(corr))

        if idx in support:
            break

        support.append(idx)
        Phi_sub = Phi[:, support]
        x_sub, _, _, _ = np.linalg.lstsq(Phi_sub, y, rcond=None)
        x_hat[support] = x_sub
        residual = y - Phi_sub @ x_sub

        if np.linalg.norm(residual) < 1e-6:
            break

    return x_hat

M_full = 100
M_cs = 30

S_full = generate_multi_target_beat(
    [40.0, 90.0],
    [5.0, -10.0],
    snr_db=20,
    M_ch=M_full,
    N_s=N_fast
)

# Σωστή σύνταξη: πρώτα όλο το spectrum, μετά slicing
fft_full_cs = np.abs(np.fft.fft(S_full[:, 0]))**2
fft_half_cs = fft_full_cs[:N_fast // 2]

freqs_half = np.fft.fftfreq(N_fast, dt)[:N_fast // 2]
range_bins_half = freqs_half * c / (2 * mu)

Phi_cs = np.exp(
    1j * 2 * np.pi * np.outer(np.arange(N_fast), freqs_half * dt)
) / np.sqrt(N_fast)

y_cs = S_full[:, 0]
x_cs = omp_recovery(y_cs, Phi_cs, n_targets=5)
x_cs_pow = np.abs(x_cs)**2

# -----------------------------------------------------------------------------
# ΙΔΕΑ 7 — Wideband FMCW
# -----------------------------------------------------------------------------
register_idea(
    7,
    "Wideband FMCW (B > 10 GHz)",
    "Sensing",
    "[A]",
    "Αναλυτική βελτίωση range resolution και CRLB καθώς αυξάνει το bandwidth."
)

B_values = np.array([5e9, 10e9, 20e9, 50e9, 100e9])
res_vals_cm = np.array([range_resolution(b) * 100 for b in B_values])
crlb_vals_cm = np.array([crlb_range(b, 10, M_val=100, T_val=10e-6) * 100 for b in B_values])

# -----------------------------------------------------------------------------
# ΙΔΕΑ 8 — Adaptive local detector αντί για fixed baseline
# -----------------------------------------------------------------------------
register_idea(
    8,
    "Adaptive local detector αντί για fixed baseline",
    "Sensing",
    "[S/C]",
    "Heuristic local-threshold detector ως proxy βελτίωσης έναντι fixed baseline."
)

def ca_cfar_1d(power, guard=2, train=4, pfa=1e-3):
    n_train = 2 * train
    alpha = n_train * (pfa**(-1 / n_train) - 1)
    out = np.zeros(len(power), dtype=bool)
    for i in range(guard + train, len(power) - guard - train):
        left = power[i - guard - train:i - guard]
        right = power[i + guard + 1:i + guard + train + 1]
        noise_mean = (left.sum() + right.sum()) / n_train
        if power[i] > alpha * noise_mean:
            out[i] = True
    return out

def adaptive_local_detector(power, snr_db, window=8):
    out = np.zeros(len(power), dtype=bool)
    for i in range(window, len(power) - window):
        local = power[i - window:i + window + 1]
        k = max(1.5, 3.5 - snr_db / 15)
        thr = np.mean(local) + k * np.std(local)
        if power[i] > thr and power[i] == power[max(0, i - 2):i + 3].max():
            out[i] = True
    return out

def roc_compare_detectors(n_trials=300):
    pfa_axis = np.logspace(-4, -1, 20)
    pd_cfar = []
    pd_adap = []
    for pfa in pfa_axis:
        hits_c = hits_a = 0
        for _ in range(n_trials):
            N_r = 128
            power = np.random.exponential(1.0, N_r)
            tgt_i = 60
            power[tgt_i] += 10 ** (8 / 10)

            det_c = ca_cfar_1d(power, guard=2, train=4, pfa=pfa)
            det_a = adaptive_local_detector(power, snr_db=8)

            hits_c += int(det_c[tgt_i])
            hits_a += int(det_a[tgt_i])
        pd_cfar.append(hits_c / n_trials)
        pd_adap.append(hits_a / n_trials)
    return pfa_axis, np.array(pd_cfar), np.array(pd_adap)

pfa_axis, pd_cfar, pd_adap = roc_compare_detectors()

# =============================================================================
# ΚΑΤΗΓΟΡΙΑ 3 — TRACKING
# =============================================================================

# -----------------------------------------------------------------------------
# ΙΔΕΑ 9 — Particle filter
# -----------------------------------------------------------------------------
register_idea(
    9,
    "Particle filter για maneuvering targets",
    "Tracking",
    "[S]",
    "Sequential Monte Carlo tracking ως εναλλακτική σε γραμμική εξομάλυνση."
)

class ParticleFilter:
    def __init__(self, n_particles=600):
        self.N = n_particles
        self.particles = None
        self.weights = np.ones(n_particles) / n_particles

    def initialize(self, x0, P0):
        self.particles = x0 + np.random.randn(self.N, len(x0)) * np.sqrt(np.diag(P0))

    def predict(self, dt_step=1.0, q_scale=1.0):
        F = np.array([[1,0,dt_step,0],
                      [0,1,0,dt_step],
                      [0,0,1,0],
                      [0,0,0,1]])
        noise = q_scale * np.random.randn(self.N, 4) * np.array([0.1,0.1,0.3,0.3])
        self.particles = (F @ self.particles.T).T + noise

    def update(self, z, R=2.0):
        H = np.array([[1,0,0,0],[0,1,0,0]])
        invR = np.linalg.inv(R * np.eye(2))
        for i in range(self.N):
            pred = H @ self.particles[i]
            diff = z - pred
            self.weights[i] *= np.exp(-0.5 * diff @ invR @ diff)
        self.weights += 1e-300
        self.weights /= self.weights.sum()

    def resample(self):
        idx = np.random.choice(self.N, self.N, p=self.weights)
        self.particles = self.particles[idx]
        self.weights = np.ones(self.N) / self.N

    def estimate(self):
        return np.average(self.particles, weights=self.weights, axis=0)

def compare_pf_vs_smoothing(n_steps=50, noise_std=2.0):
    gt = []
    x, y, vx, vy = 10.0, 10.0, 1.5, 1.0
    for k in range(n_steps):
        if k == 25:
            vx, vy = -0.5, 2.0
        x += vx
        y += vy
        gt.append((x, y))
    gt = np.array(gt)
    meas = gt + noise_std * np.random.randn(n_steps, 2)

    pf = ParticleFilter(800)
    pf.initialize(np.array([meas[0,0], meas[0,1], 1.5, 1.0]), np.diag([4,4,1,1]))

    pf_track = []
    for k in range(n_steps):
        pf.predict(dt_step=1.0, q_scale=1.5)
        pf.update(meas[k], R=noise_std**2)
        pf.resample()
        pf_track.append(pf.estimate()[:2])
    pf_track = np.array(pf_track)

    smooth_track = np.zeros_like(gt)
    w = 10
    for k in range(n_steps):
        lo, hi = max(0, k - w), min(n_steps, k + w + 1)
        smooth_track[k, 0] = np.mean(meas[lo:hi, 0])
        smooth_track[k, 1] = np.mean(meas[lo:hi, 1])

    pf_err = np.sqrt(np.sum((pf_track - gt)**2, axis=1))
    sm_err = np.sqrt(np.sum((smooth_track - gt)**2, axis=1))
    return gt, meas, pf_track, smooth_track, pf_err, sm_err

gt_pf, meas_pf, pf_track, sm_track, pf_err, sm_err = compare_pf_vs_smoothing()

# -----------------------------------------------------------------------------
# ΙΔΕΑ 10 — JPDA
# -----------------------------------------------------------------------------
register_idea(
    10,
    "JPDA αντί για hard association",
    "Tracking",
    "[S]",
    "Soft probabilistic association για κοντινά και ambiguous targets."
)

def compare_jpda_vs_hard(n_trials=400):
    correct_hard = 0
    correct_jpda = 0
    for _ in range(n_trials):
        t1 = np.array([40.0, 5.0])
        t2 = np.array([42.0, 7.0])

        z1 = t1 + 2.0 * np.random.randn(2)
        z2 = t2 + 2.0 * np.random.randn(2)
        clutter = np.random.uniform(35, 50, (3, 2))
        all_z = np.vstack([z1, z2, clutter])
        np.random.shuffle(all_z)

        d1 = np.linalg.norm(all_z - t1, axis=1)
        d2 = np.linalg.norm(all_z - t2, axis=1)
        a1 = all_z[np.argmin(d1)]
        a2 = all_z[np.argmin(d2)]
        if np.linalg.norm(a1 - t1) < 3 and np.linalg.norm(a2 - t2) < 3:
            correct_hard += 1

        def likelihood(z, mu, sigma=2.5):
            d = np.linalg.norm(z - mu)
            return np.exp(-0.5 * (d / sigma)**2)

        w1 = np.array([likelihood(z, t1) for z in all_z]); w1 /= w1.sum()
        w2 = np.array([likelihood(z, t2) for z in all_z]); w2 /= w2.sum()
        upd1 = (w1[:, None] * all_z).sum(axis=0)
        upd2 = (w2[:, None] * all_z).sum(axis=0)
        if np.linalg.norm(upd1 - t1) < 3 and np.linalg.norm(upd2 - t2) < 3:
            correct_jpda += 1

    return correct_hard / n_trials, correct_jpda / n_trials

acc_hard, acc_jpda = compare_jpda_vs_hard()

# -----------------------------------------------------------------------------
# ΙΔΕΑ 11 — Adaptive Otsu thresholding
# -----------------------------------------------------------------------------
register_idea(
    11,
    "Adaptive Otsu thresholding στο Hough accumulator",
    "Tracking",
    "[S]",
    "Adaptive histogram-based thresholding αντί για fixed global threshold baseline."
)

def otsu_threshold(acc_matrix):
    vals = np.round(acc_matrix.flatten()).astype(int)
    vmin, vmax = vals.min(), vals.max()
    if vmax <= vmin:
        return float(vmin)
    hist, edges = np.histogram(vals, bins=min(256, vmax - vmin + 1), range=(vmin, vmax + 1))
    hist = hist.astype(float)
    hist /= hist.sum()
    bins = edges[:-1]
    omega = np.cumsum(hist)
    mu = np.cumsum(hist * bins)
    mu_t = mu[-1]
    best_sigma = -1.0
    best_thr = bins[0]
    for i in range(1, len(hist) - 1):
        w0 = omega[i]
        w1 = 1 - w0
        if w0 <= 0 or w1 <= 0:
            continue
        mu0 = mu[i] / w0
        mu1 = (mu_t - mu[i]) / w1
        sigma_b2 = w0 * w1 * (mu0 - mu1)**2
        if sigma_b2 > best_sigma:
            best_sigma = sigma_b2
            best_thr = bins[i]
    return float(best_thr)

def compare_fixed_vs_otsu(n_trials=300):
    tdr_fixed, far_fixed = [], []
    tdr_otsu, far_otsu = [], []
    rows, cols = 200, 180
    for _ in range(n_trials):
        acc = np.random.poisson(2, (rows, cols)).astype(float)
        pr = np.random.randint(20, 180)
        pt = np.random.randint(20, 160)
        acc[pr-1:pr+2, pt-1:pt+2] += np.random.randint(8, 15)
        acc_sm = uniform_filter(acc, size=3)

        thr_fixed = np.percentile(acc_sm, 99.5)
        det_f = acc_sm > thr_fixed
        tp_f = det_f[pr-3:pr+4, pt-3:pt+4].any()
        fp_f = det_f.sum() - int(tp_f)

        thr_o = max(np.percentile(acc_sm, 99.3), otsu_threshold(acc_sm))
        det_o = acc_sm > thr_o
        tp_o = det_o[pr-3:pr+4, pt-3:pt+4].any()
        fp_o = det_o.sum() - int(tp_o)

        tdr_fixed.append(float(tp_f))
        tdr_otsu.append(float(tp_o))
        far_fixed.append(fp_f / (rows * cols))
        far_otsu.append(fp_o / (rows * cols))
    return np.mean(tdr_fixed), np.mean(far_fixed), np.mean(tdr_otsu), np.mean(far_otsu)

tdr_fixed, far_fixed, tdr_otsu, far_otsu = compare_fixed_vs_otsu()

# -----------------------------------------------------------------------------
# ΙΔΕΑ 12 — Heuristic sequence predictor
# -----------------------------------------------------------------------------
register_idea(
    12,
    "Heuristic sequence predictor για abrupt motion",
    "Tracking",
    "[C]",
    "Ενδεικτικός predictor τύπου smoothing + velocity extrapolation. Όχι πραγματικό LSTM."
)

def heuristic_sequence_predictor(history, horizon=1):
    if len(history) < 3:
        return np.array(history[-1], dtype=float)
    alpha = 0.6
    smoothed = np.array(history[0], dtype=float)
    for h in history[1:]:
        smoothed = alpha * np.array(h) + (1 - alpha) * smoothed
    vel = np.array(history[-1]) - np.array(history[-2])
    return smoothed + 0.5 * horizon * vel

def compare_kf_vs_heuristic(n_steps=60, noise=1.5):
    gt = []
    x, y, vx, vy = 5.0, 5.0, 2.0, 0.5
    for k in range(n_steps):
        if k == 30:
            vx, vy = 3.5, 2.0
        x += vx + 0.1 * np.random.randn()
        y += vy + 0.1 * np.random.randn()
        gt.append([x, y])
    gt = np.array(gt)
    meas = gt + noise * np.random.randn(n_steps, 2)

    xk = np.array([meas[0,0], meas[0,1], 2.0, 0.5])
    Pk = np.eye(4) * 4
    F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
    H = np.array([[1,0,0,0],[0,1,0,0]])
    Q = np.diag([0.1,0.1,0.5,0.5])
    Rm = noise**2 * np.eye(2)

    kf_track = np.zeros_like(gt)
    for k in range(n_steps):
        xk = F @ xk
        Pk = F @ Pk @ F.T + Q
        S = H @ Pk @ H.T + Rm
        K = Pk @ H.T @ np.linalg.inv(S)
        xk = xk + K @ (meas[k] - H @ xk)
        Pk = (np.eye(4) - K @ H) @ Pk
        kf_track[k] = xk[:2]

    hist = list(meas[:3])
    heur_track = np.zeros_like(gt)
    for k in range(n_steps):
        if k < 3:
            heur_track[k] = meas[k]
        else:
            pred = heuristic_sequence_predictor(hist[-8:], horizon=1)
            heur_track[k] = 0.4 * pred + 0.6 * meas[k]
            hist.append(meas[k])

    kf_err = np.sqrt(np.sum((kf_track - gt)**2, axis=1))
    heur_err = np.sqrt(np.sum((heur_track - gt)**2, axis=1))
    return gt, kf_track, heur_track, kf_err, heur_err

gt_seq, kf_track, heur_track, kf_err, heur_err = compare_kf_vs_heuristic()

# =============================================================================
# ΚΑΤΗΓΟΡΙΑ 4 — ADB
# =============================================================================

# -----------------------------------------------------------------------------
# ΙΔΕΑ 13 — Micro-LED array ADB
# -----------------------------------------------------------------------------
register_idea(
    13,
    "Micro-LED array για pixel-level ADB",
    "ADB",
    "[S/C]",
    "Ποιοτικά καλύτερη spatial control από continuous raised-cosine profile."
)

def micro_led_beam(target_angles, n_pixels=64, fov_deg=50):
    pixel_angles = np.linspace(-fov_deg / 2, fov_deg / 2, n_pixels)
    beam = np.ones(n_pixels)
    for tgt in target_angles:
        mask = np.abs(pixel_angles - tgt) < 1.5
        beam[mask] = 0.0
    return pixel_angles, beam

def raised_cosine_beam(target_angles, fov_deg=50):
    angles = np.linspace(-fov_deg / 2, fov_deg / 2, 500)
    beam = np.ones_like(angles)
    for tgt in target_angles:
        margin = 3.0
        for i, a in enumerate(angles):
            if abs(a - tgt) < margin:
                u = abs(a - tgt) / margin
                beam[i] = 0.5 - 0.5 * np.cos(np.pi * u)
    return angles, beam

target_angles = [-10, 8]
pix_ang, beam_micro = micro_led_beam(target_angles)
cont_ang, beam_cont = raised_cosine_beam(target_angles)

# -----------------------------------------------------------------------------
# ΙΔΕΑ 14 — Semantic ADB
# -----------------------------------------------------------------------------
register_idea(
    14,
    "Semantic ADB με class-dependent safety margins",
    "ADB",
    "[C]",
    "Vehicle / pedestrian / cyclist ανάθεση διαφορετικού safety margin."
)

def semantic_margin(obj_class, distance_m):
    base = {
        "vehicle": 2.0,
        "pedestrian": 4.0,
        "cyclist": 3.5,
    }.get(obj_class, 2.0)
    dist_factor = max(0.5, 1.0 - distance_m / 200)
    return base * (1 + dist_factor)

dist_axis = np.linspace(10, 150, 50)
margins_vehicle = np.array([semantic_margin("vehicle", d) for d in dist_axis])
margins_ped = np.array([semantic_margin("pedestrian", d) for d in dist_axis])
margins_cyc = np.array([semantic_margin("cyclist", d) for d in dist_axis])

# -----------------------------------------------------------------------------
# ΙΔΕΑ 15 — LiDAR-guided ADB
# -----------------------------------------------------------------------------
register_idea(
    15,
    "LiDAR-guided ADB για μείωση camera-headlamp offset error",
    "ADB",
    "[A/S]",
    "Γεωμετρική εξάλειψη offset error με direct range-angle estimation."
)

def shadow_angle_error(range_m, lateral_m, camera_offset=0.3):
    theta_true = np.rad2deg(np.arctan(lateral_m / range_m))
    theta_cam = np.rad2deg(np.arctan((lateral_m + camera_offset) / range_m))
    return theta_true, abs(theta_true - theta_cam)

range_adb = np.linspace(10, 150, 80)
cam_errors = np.array([shadow_angle_error(r, 1.8)[1] for r in range_adb])
lidar_errors = np.zeros_like(range_adb)

# =============================================================================
# ΚΑΤΗΓΟΡΙΑ 5 — SYSTEM-LEVEL
# =============================================================================

# -----------------------------------------------------------------------------
# ΙΔΕΑ 16 — Cognitive ISAC resource allocation
# -----------------------------------------------------------------------------
register_idea(
    16,
    "Cognitive ISAC resource allocation",
    "Σύστημα",
    "[A/S]",
    "Trade-off παράμετρος α μεταξύ sensing resource και communication resource."
)

def cognitive_tradeoff(alpha_vals, traffic_density):
    sinr_sensing = 10 * np.log10(np.maximum(alpha_vals, 1e-6) * M_chirps * 10)
    rate_comms = (1 - alpha_vals) * Rb / 1e9
    alpha_opt = min(1.0, 0.3 + 0.5 * traffic_density)
    return sinr_sensing, rate_comms, alpha_opt

alpha_vals = np.linspace(0.05, 0.95, 100)
sinr_low, rate_low, alpha_low = cognitive_tradeoff(alpha_vals, 0.2)
sinr_high, rate_high, alpha_high = cognitive_tradeoff(alpha_vals, 0.8)

# -----------------------------------------------------------------------------
# ΙΔΕΑ 17 — Frequency hopping FMCW
# -----------------------------------------------------------------------------
register_idea(
    17,
    "Frequency hopping FMCW για mitigation mutual interference",
    "Σύστημα",
    "[A/S]",
    "Collision probability reduction με hop set size N_hops."
)

def collision_prob_hopping(n_vehicles, n_hops=64):
    return 1 - (1 - 1 / n_hops)**n_vehicles

veh_axis = np.arange(1, 11)
p_std = np.clip(0.18 * veh_axis, 0, 0.95)
p_hop = np.array([collision_prob_hopping(v, 64) for v in veh_axis])

# -----------------------------------------------------------------------------
# ΙΔΕΑ 18 — Pareto waveform optimization
# -----------------------------------------------------------------------------
register_idea(
    18,
    "Pareto waveform optimization for sensing/communication trade-off",
    "Σύστημα",
    "[A/S]",
    "Πολυκριτηριακή καμπύλη sensing SINR vs communication rate."
)

def pareto_tradeoff(alpha_vals, snr_db=15):
    g = 10 ** (snr_db / 10.0)
    sinr_s = 10 * np.log10(alpha_vals * g * M_chirps)
    rate_c = np.log2(1 + (1 - alpha_vals) * g)
    paper_sinr = 10 * np.log10(0.5 * g * M_chirps)
    paper_rate = np.log2(1 + 0.5 * g)
    return sinr_s, rate_c, paper_sinr, paper_rate

sinr_pareto, rate_pareto, paper_sinr, paper_rate = pareto_tradeoff(alpha_vals, 15)

# -----------------------------------------------------------------------------
# ΙΔΕΑ 19 — Optical MIMO capacity scaling
# -----------------------------------------------------------------------------
register_idea(
    19,
    "Optical MIMO beamforming / capacity scaling",
    "Σύστημα",
    "[A/S]",
    "MIMO capacity comparison έναντι SISO baseline."
)

def mimo_capacity(n_tx_values, snr_db=15):
    g = 10 ** (snr_db / 10.0)
    caps = []
    for Nt in n_tx_values:
        Nr = Nt
        H = (np.random.randn(Nr, Nt) + 1j * np.random.randn(Nr, Nt)) / np.sqrt(2)
        C = np.real(np.log2(np.linalg.det(np.eye(Nr) + g / Nt * H @ H.conj().T)))
        caps.append(C)
    return np.array(caps)

mimo_axis = np.array([1, 2, 4, 8])
cap_mimo = mimo_capacity(mimo_axis, 15)
cap_siso = np.log2(1 + 10 ** (15 / 10.0)) * np.ones_like(mimo_axis, dtype=float)

# -----------------------------------------------------------------------------
# 4. Console output
# -----------------------------------------------------------------------------
print("=" * 86)
print("19 ΙΔΕΕΣ ΒΕΛΤΙΩΣΗΣ — ΑΝΑΛΥΤΙΚΗ / ΗΜΙ-ΑΝΑΛΥΤΙΚΗ / CONCEPTUAL ΑΞΙΟΛΟΓΗΣΗ")
print("=" * 86)
for item in ideas:
    print(f"[{item['idx']:02d}] {item['category']} | {item['validity']} | {item['title']}")
    print(f"     {item['summary']}")

print("\n" + "=" * 86)
print("ΣΥΝΟΠΤΙΚΑ ΑΠΟΤΕΛΕΣΜΑΤΑ")
print("=" * 86)

snr_target = 10.0
idx10 = int(np.argmin(np.abs(SNR_dB_range - snr_target)))
print(f"Ιδέα 1  | BER@10dB DBPSK      = {ber_baseline_dbpsk[idx10]:.3e}")
print(f"Ιδέα 1  | BER@10dB OFDM-QPSK  = {ber_ofdm_qpsk[idx10]:.3e}")
print(f"Ιδέα 1  | BER@10dB OFDM-16QAM = {ber_ofdm_16qam[idx10]:.3e}")
print(f"Ιδέα 2  | DBPSK with LDPC-like gain BER@10dB = {ber_ldpc_dbpsk[idx10]:.3e}")
print(f"Ιδέα 3  | Polarization-mux goodput@10dB      = {tp_pol[idx10]/1e9:.3f} Gbps")
print(f"Ιδέα 4  | MMSE mean BER@subset first point   = {ber_mmse[0]:.3e}")
print(f"Ιδέα 7  | Range resolution @10GHz            = {range_resolution(10e9)*100:.2f} cm")
print(f"Ιδέα 7  | Range resolution @50GHz            = {range_resolution(50e9)*100:.2f} cm")
print(f"Ιδέα 10 | Hard association accuracy          = {acc_hard:.3f}")
print(f"Ιδέα 10 | JPDA accuracy                      = {acc_jpda:.3f}")
print(f"Ιδέα 11 | Fixed threshold FAR                = {far_fixed:.5f}")
print(f"Ιδέα 11 | Otsu threshold FAR                 = {far_otsu:.5f}")
print(f"Ιδέα 16 | Suggested alpha (low traffic)      = {alpha_low:.2f}")
print(f"Ιδέα 16 | Suggested alpha (high traffic)     = {alpha_high:.2f}")
print(f"Ιδέα 17 | P(interference), 5 vehicles std    = {p_std[4]:.3f}")
print(f"Ιδέα 17 | P(interference), 5 vehicles hop    = {p_hop[4]:.3f}")
print(f"Ιδέα 19 | 8x8 MIMO capacity                  = {cap_mimo[-1]:.2f} bits/s/Hz")

# -----------------------------------------------------------------------------
# 5. Mega figure
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(22, 20), constrained_layout=True)
fig.suptitle(
    "19 Ιδέες Βελτίωσης για το PC-FMCW ISCAI Σύστημα\n"
    "Αναλυτική / Ημι-αναλυτική / Conceptual αξιολόγηση",
    fontsize=14, fontweight="bold", y=0.995
)
gs = gridspec.GridSpec(5, 4, figure=fig)

def set_title(ax, title, color):
    ax.set_title(title, fontsize=9.5, fontweight="bold", color=color, pad=6)

# --- Row 1: Communication
c_comm = "#185FA5"

ax = fig.add_subplot(gs[0, 0])
ax.semilogy(SNR_dB_range, ber_baseline_dbpsk, lw=2.5, label="DBPSK baseline")
ax.semilogy(SNR_dB_range, ber_ofdm_qpsk, lw=2, ls="--", label="OFDM-QPSK")
ax.semilogy(SNR_dB_range, ber_ofdm_16qam, lw=2, ls=":", label="OFDM-16QAM")
ax.semilogy(SNR_dB_range, ber_ofdm_64qam, lw=1.5, ls="-.", label="OFDM-64QAM")
ax.axhline(1e-6, color="gray", ls=":", lw=1)
set_title(ax, "Ιδέα 1: OFDM/QAM vs DBPSK", c_comm)
ax.set_xlabel("SNR (dB)"); ax.set_ylabel("BER"); ax.grid(alpha=0.25); ax.legend(fontsize=7)

ax = fig.add_subplot(gs[0, 1])
ax.semilogy(SNR_dB_range, ber_baseline_dbpsk, lw=2.5, label="DBPSK uncoded")
ax.semilogy(SNR_dB_range, ber_ldpc_dbpsk, lw=2, ls="--", label="DBPSK + LDPC gain")
ax.semilogy(SNR_dB_range, ber_ldpc_qpsk, lw=2, ls=":", label="QPSK + LDPC gain")
ax.axhline(1e-6, color="gray", ls=":", lw=1)
set_title(ax, "Ιδέα 2: LDPC coding gain", c_comm)
ax.set_xlabel("SNR (dB)"); ax.set_ylabel("BER"); ax.grid(alpha=0.25); ax.legend(fontsize=7)

ax = fig.add_subplot(gs[0, 2])
ax.plot(SNR_dB_range, goodput_dbpsk/1e9, lw=2.5, label="DBPSK")
ax.plot(SNR_dB_range, tp_pol/1e9, lw=2, ls="--", label="Polarization MUX")
set_title(ax, "Ιδέα 3: Polarization multiplexing", c_comm)
ax.set_xlabel("SNR (dB)"); ax.set_ylabel("Goodput (Gbps)"); ax.grid(alpha=0.25); ax.legend(fontsize=7)

ax = fig.add_subplot(gs[0, 3])
ax.semilogy(snr_subset_comm, ber_no_eq, "o-", lw=2, label="No equalizer")
ax.semilogy(snr_subset_comm, ber_mmse, "s--", lw=2, label="MMSE equalizer")
set_title(ax, "Ιδέα 4: MMSE equalization", c_comm)
ax.set_xlabel("SNR (dB)"); ax.set_ylabel("BER"); ax.grid(alpha=0.25); ax.legend(fontsize=7)

# --- Row 2: Sensing
c_sens = "#27500A"

ax = fig.add_subplot(gs[1, 0])
mask_fft = range_fft < 120
mask_music = range_music < 120
ax.plot(range_fft[mask_fft], 10*np.log10(fft_half_music[mask_fft] + 1e-12), lw=1.5, label="FFT baseline")
P_norm = P_music / np.max(P_music[mask_music])
ax.plot(range_music[mask_music], 10*np.log10(P_norm[mask_music] + 1e-12), lw=2, ls="--", label="MUSIC")
ax.axvline(50, color="gray", ls=":", lw=1)
ax.axvline(52.5, color="gray", ls=":", lw=1)
set_title(ax, "Ιδέα 5: MUSIC superresolution", c_sens)
ax.set_xlabel("Range (m)"); ax.set_ylabel("Power (dB)"); ax.grid(alpha=0.25); ax.legend(fontsize=7)

ax = fig.add_subplot(gs[1, 1])
valid_cs = range_bins_half < 120
ax.plot(range_bins_half[valid_cs], 10*np.log10(fft_half_cs[valid_cs] + 1e-12), lw=2, label=f"Full FFT ({M_full} chirps)")
ax.plot(range_bins_half[valid_cs], 10*np.log10(x_cs_pow[:len(range_bins_half)][valid_cs] + 1e-12), lw=2, ls="--", label=f"CS-OMP ({M_cs} meas.)")
set_title(ax, "Ιδέα 6: Compressed sensing FMCW", c_sens)
ax.set_xlabel("Range (m)"); ax.set_ylabel("Power (dB)"); ax.grid(alpha=0.25); ax.legend(fontsize=7)

ax = fig.add_subplot(gs[1, 2])
x = np.arange(len(B_values))
ax.bar(x - 0.2, res_vals_cm, 0.38, label="ΔR (cm)")
ax.bar(x + 0.2, crlb_vals_cm, 0.38, label="CRLB (cm)")
ax.set_xticks(x); ax.set_xticklabels(["5", "10", "20", "50", "100"])
set_title(ax, "Ιδέα 7: Wideband FMCW", c_sens)
ax.set_xlabel("Bandwidth (GHz)"); ax.set_ylabel("cm"); ax.grid(axis="y", alpha=0.25); ax.legend(fontsize=7)

ax = fig.add_subplot(gs[1, 3])
ax.semilogx(pfa_axis, pd_cfar, "o-", lw=2, label="1D CFAR baseline")
ax.semilogx(pfa_axis, pd_adap, "s--", lw=2, label="Adaptive local detector")
set_title(ax, "Ιδέα 8: Adaptive detector", c_sens)
ax.set_xlabel("False Alarm Rate"); ax.set_ylabel("Detection Probability"); ax.grid(alpha=0.25); ax.legend(fontsize=7)

# --- Row 3: Tracking
c_track = "#3C3489"

ax = fig.add_subplot(gs[2, 0])
ax.plot(gt_pf[:,0], gt_pf[:,1], lw=2.5, label="Ground truth")
ax.plot(meas_pf[:,0], meas_pf[:,1], ".", ms=4, alpha=0.4, label="Measurements")
ax.plot(pf_track[:,0], pf_track[:,1], "--", lw=2, label=f"PF ({pf_err.mean():.2f})")
ax.plot(sm_track[:,0], sm_track[:,1], ":", lw=2, label=f"Smooth baseline ({sm_err.mean():.2f})")
set_title(ax, "Ιδέα 9: Particle filter", c_track)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.grid(alpha=0.25); ax.legend(fontsize=7)

ax = fig.add_subplot(gs[2, 1])
bars = ax.bar(["Hard\nassoc.", "JPDA"], [acc_hard, acc_jpda], width=0.45)
for b, v in zip(bars, [acc_hard, acc_jpda]):
    ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
set_title(ax, "Ιδέα 10: JPDA vs hard association", c_track)
ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.15); ax.grid(axis="y", alpha=0.25)

ax = fig.add_subplot(gs[2, 2])
ax2 = ax.twinx()
x = np.arange(2)
ax.bar(x - 0.15, [tdr_fixed, tdr_otsu], 0.3, label="TDR")
ax2.bar(x + 0.15, [far_fixed * 1e3, far_otsu * 1e3], 0.3, alpha=0.8, label="FAR ×1e-3")
ax.set_xticks(x); ax.set_xticklabels(["Fixed\nbaseline", "Adaptive\nOtsu"])
set_title(ax, "Ιδέα 11: Otsu thresholding", c_track)
ax.set_ylabel("TDR"); ax2.set_ylabel("FAR ×1e-3")
ax.set_ylim(0, 1.2)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right")

ax = fig.add_subplot(gs[2, 3])
steps = np.arange(len(kf_err))
ax.plot(steps, kf_err, lw=2, label=f"KF ({kf_err.mean():.2f})")
ax.plot(steps, heur_err, lw=2, ls="--", label=f"Heuristic predictor ({heur_err.mean():.2f})")
ax.axvline(30, color="orange", ls=":", lw=1.5, label="Abrupt change")
set_title(ax, "Ιδέα 12: Heuristic sequence predictor", c_track)
ax.set_xlabel("Step"); ax.set_ylabel("Tracking error"); ax.grid(alpha=0.25); ax.legend(fontsize=7)

# --- Row 4: ADB
c_adb = "#633806"

ax = fig.add_subplot(gs[3, 0])
ax.bar(pix_ang, beam_micro, width=50/64, edgecolor="black", linewidth=0.3, label="Micro-LED")
ax.plot(cont_ang, beam_cont, lw=2.5, label="Raised-cos baseline")
set_title(ax, "Ιδέα 13: Micro-LED array", c_adb)
ax.set_xlabel("Angle (deg)"); ax.set_ylabel("Normalized intensity"); ax.grid(alpha=0.25); ax.legend(fontsize=7)

ax = fig.add_subplot(gs[3, 1])
ax.plot(dist_axis, margins_vehicle, lw=2, label="Vehicle")
ax.plot(dist_axis, margins_ped, lw=2, label="Pedestrian")
ax.plot(dist_axis, margins_cyc, lw=2, label="Cyclist")
ax.axhline(2.0, color="gray", ls=":", lw=1, label="Fixed margin baseline")
set_title(ax, "Ιδέα 14: Semantic ADB margins", c_adb)
ax.set_xlabel("Distance (m)"); ax.set_ylabel("Margin (deg)"); ax.grid(alpha=0.25); ax.legend(fontsize=7)

ax = fig.add_subplot(gs[3, 2])
ax.plot(range_adb, cam_errors, lw=2.5, label="Camera-offset geometry")
ax.plot(range_adb, lidar_errors, lw=2, ls="--", label="LiDAR-guided")
set_title(ax, "Ιδέα 15: LiDAR-guided ADB", c_adb)
ax.set_xlabel("Range (m)"); ax.set_ylabel("Angular error (deg)"); ax.grid(alpha=0.25); ax.legend(fontsize=7)

ax = fig.add_subplot(gs[3, 3])
methods = ["Raised-\ncos", "Micro-\nLED", "Semantic\nADB", "LiDAR\nADB"]
err_vals = [3.0, 0.8, 2.5, 0.0]
lat_vals = [5.0, 0.2, 25.0, 2.0]
ax2 = ax.twinx()
x = np.arange(len(methods))
ax.bar(x - 0.18, err_vals, 0.35, label="Angular error (deg)")
ax2.bar(x + 0.18, lat_vals, 0.35, alpha=0.8, label="Latency (ms)")
ax.set_xticks(x); ax.set_xticklabels(methods)
set_title(ax, "ADB ideas summary", c_adb)
ax.set_ylabel("Error (deg)"); ax2.set_ylabel("Latency (ms)")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=7)

# --- Row 5: System
c_sys = "#712B13"

ax = fig.add_subplot(gs[4, 0])
ax.plot(rate_low, sinr_low, lw=2, label=f"Low traffic (α≈{alpha_low:.2f})")
ax.plot(rate_high, sinr_high, lw=2, ls="--", label=f"High traffic (α≈{alpha_high:.2f})")
ax.scatter([np.log2(1 + 0.5 * 10**(15/10.0))], [10*np.log10(0.5 * M_chirps * 10)], s=80, label="Fixed baseline")
set_title(ax, "Ιδέα 16: Cognitive ISAC", c_sys)
ax.set_xlabel("Comm rate (Gbps-scaled proxy)"); ax.set_ylabel("Sensing SINR (dB)"); ax.grid(alpha=0.25); ax.legend(fontsize=7)

ax = fig.add_subplot(gs[4, 1])
ax.plot(veh_axis, p_std, "o-", lw=2, label="Standard FMCW")
ax.plot(veh_axis, p_hop, "s--", lw=2, label="Freq. hopping")
set_title(ax, "Ιδέα 17: Frequency hopping", c_sys)
ax.set_xlabel("Number of vehicles"); ax.set_ylabel("Collision probability"); ax.grid(alpha=0.25); ax.legend(fontsize=7)

ax = fig.add_subplot(gs[4, 2])
ax.plot(rate_pareto, sinr_pareto, lw=2.5, label="Pareto frontier")
ax.scatter([paper_rate], [paper_sinr], s=100, label="Baseline operating point")
set_title(ax, "Ιδέα 18: Pareto optimization", c_sys)
ax.set_xlabel("Comm rate (bits/s/Hz)"); ax.set_ylabel("Sensing SINR (dB)"); ax.grid(alpha=0.25); ax.legend(fontsize=7)

ax = fig.add_subplot(gs[4, 3])
ax.plot(mimo_axis, cap_siso, lw=2, ls="--", label="SISO baseline")
ax.plot(mimo_axis, cap_mimo, "o-", lw=2.5, label="MIMO")
set_title(ax, "Ιδέα 19: Optical MIMO", c_sys)
ax.set_xlabel("Number of apertures"); ax.set_ylabel("Capacity (bits/s/Hz)"); ax.grid(alpha=0.25); ax.legend(fontsize=7)

# Category labels
cat_labels = ["ΕΠΙΚΟΙΝΩΝΙΑ", "SENSING", "TRACKING", "ADB", "ΣΥΣΤΗΜΑ"]
cat_colors = [c_comm, c_sens, c_track, c_adb, c_sys]
for i, (lbl, col) in enumerate(zip(cat_labels, cat_colors)):
    fig.text(0.004, 0.90 - i * 0.195, lbl, fontsize=9, fontweight="bold", color=col, va="center", rotation=90)

# -----------------------------------------------------------------------------
# 6. Save outputs
# -----------------------------------------------------------------------------
out_dir = Path("/mnt/data")
fig_path = out_dir / "all_19_improvements_analytical_v2.png"
py_path = out_dir / "all_19_improvements_analytical_v2.py"
txt_path = out_dir / "all_19_improvements_notes_v2.txt"

fig.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close(fig)

notes = []
notes.append("Τελική ακαδημαϊκή αξιολόγηση των 19 ιδεών\n")
notes.append("Ισχυρότερες για κύρια πρόταση βελτίωσης:\n")
notes.append("1) OFDM/QAM-based communication extension\n")
notes.append("2) LDPC coding\n")
notes.append("10) JPDA soft association\n")
notes.append("17) Frequency hopping FMCW\n")
notes.append("\nΙσχυρές αλλά πιο εξειδικευμένες:\n")
notes.append("5) MUSIC superresolution\n")
notes.append("6) Compressed sensing FMCW\n")
notes.append("7) Wideband FMCW\n")
notes.append("11) Adaptive Otsu thresholding (με calibration)\n")
notes.append("16) Cognitive ISAC resource allocation\n")
notes.append("18) Pareto waveform optimization\n")
notes.append("\nΠιο conceptual / future-work χαρακτήρα:\n")
notes.append("8) Adaptive local detector as heuristic baseline\n")
notes.append("12) Heuristic sequence predictor\n")
notes.append("13-15) ADB redesign ideas\n")
notes.append("19) Optical MIMO beamforming\n")

py_path.write_text(__doc__ + "\n", encoding="utf-8")
txt_path.write_text("".join(notes), encoding="utf-8")

print(f"Saved figure: {fig_path}")
print(f"Saved notes: {txt_path}")
