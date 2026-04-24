import matplotlib
matplotlib.use("Agg")
# -*- coding: utf-8 -*-
"""
ΙΔΕΑ 3 — Blind Spot Coverage μέσω Cooperative Sensing
======================================================
Αμάξι Β βλέπει την τυφλή γωνία του Α και μεταδίδει τα δεδομένα.

Θεωρία
------
Κάθε αμάξι έχει τυφλές γωνίες λόγω:
  (α) γεωμετρικής occlusion (εμπόδια μπροστά/πλαϊνά)
  (β) περιορισμένο FOV (Field of View) του sensor

Αν αμάξι Β βρίσκεται σε διαφορετική θέση, βλέπει zones
που το Α δεν μπορεί. Στέλνει τα tracks μέσω V2X.

Μοντέλο
-------
- Αμάξι Α στο (0, 0), κινείται προς +x
- Αμάξι Β στο (d, lane_width), παράλληλα
- Εμπόδιο (parked truck) στο (L_obs, 0) που κρύβει πεζό
- Πεζός στο (L_obs + dx_ped, -w/2) — αόρατος στο Α, ορατός στο Β

Μετρικές
--------
- Detection delay: πόσο νωρίτερα ανιχνεύει το Β vs Α
- Warning time: χρόνος που απομένει πριν collision
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

np.random.seed(7)

# ─── Παράμετροι σκηνής ────────────────────────────────────────────────────
v_ego       = 50 / 3.6    # m/s  (50 km/h)
lane_width  = 3.5          # m
d_lateral   = lane_width   # απόσταση Β από Α (παράλληλη λωρίδα)

# Parked truck (εμπόδιο)
truck_x     = 30.0         # m ahead
truck_length= 8.0          # m
truck_width = 2.5          # m

# Πεζός
ped_x       = truck_x + truck_length + 1.0  # αμέσως μετά το truck
ped_y       = -lane_width / 2               # στη μέση της λωρίδας του Α
ped_v_y     = 1.2                           # m/s προς τη λωρίδα

# FOV (half-angle) κάθε αμαξιού
fov_half_deg = 30.0
fov_half     = np.deg2rad(fov_half_deg)

# ─── Occlusion γεωμετρία ──────────────────────────────────────────────────
def is_occluded(sensor_pos, target_pos, obstacles):
    """
    Ελέγχει αν ο sensor βλέπει τον target ή παρεμβαίνει obstacle.
    Απλοποιημένο: line-of-sight check.
    """
    sx, sy = sensor_pos
    tx, ty = target_pos
    for (ox, oy, ow, oh) in obstacles:
        # 4 corners του obstacle
        corners = [(ox, oy), (ox+ow, oy), (ox+ow, oy+oh), (ox, oy+oh)]
        # Αν line ST διέρχεται από το rectangle
        # Απλοποίηση: αν target είναι "πίσω" από obstacle σε σχέση με sensor
        if sx < ox and tx > ox:  # obstacle between sensor and target in x
            # interpolate y at obstacle x
            if ow > 0:
                t_interp = (ox - sx) / (tx - sx)
                y_at_obs = sy + t_interp * (ty - sy)
                if oy <= y_at_obs <= oy + oh:
                    return True
    return False

def in_fov(sensor_pos, target_pos, heading_angle, fov_half):
    """Ελέγχει αν target βρίσκεται εντός FOV του sensor."""
    dx = target_pos[0] - sensor_pos[0]
    dy = target_pos[1] - sensor_pos[1]
    angle_to_target = np.arctan2(dy, dx)
    angle_diff = abs(angle_to_target - heading_angle)
    if angle_diff > np.pi:
        angle_diff = 2*np.pi - angle_diff
    return angle_diff <= fov_half

# ─── Simulation χρονική εξέλιξη ───────────────────────────────────────────
dt_sim     = 0.1    # s
t_max      = 4.0    # s
time_steps = np.arange(0, t_max, dt_sim)

# Αρχικές θέσεις αμαξιών (αρνητικές — πλησιάζουν)
x0_A = -20.0
x0_B = -20.0

obstacle = [(truck_x, ped_y - truck_width/2, truck_length, truck_width)]

detected_A = []  # χρόνος ανίχνευσης από Α
detected_B = []  # χρόνος ανίχνευσης από Β

det_time_A = None
det_time_B = None

ped_positions = []
car_A_positions = []
car_B_positions = []

for t in time_steps:
    xA = x0_A + v_ego * t
    xB = x0_B + v_ego * t
    yA = 0.0
    yB = d_lateral

    # Πεζός κινείται προς τη λωρίδα
    xP = ped_x
    yP = ped_y + ped_v_y * t

    car_A_positions.append((xA, yA))
    car_B_positions.append((xB, yB))
    ped_positions.append((xP, yP))

    sensor_A = (xA, yA)
    sensor_B = (xB, yB)
    target_P = (xP, yP)

    fov_A_ok = in_fov(sensor_A, target_P, heading_angle=0, fov_half=fov_half)
    fov_B_ok = in_fov(sensor_B, target_P, heading_angle=0, fov_half=fov_half)

    occ_A = is_occluded(sensor_A, target_P, obstacle) if fov_A_ok else True
    occ_B = is_occluded(sensor_B, target_P, obstacle) if fov_B_ok else True

    vis_A = fov_A_ok and not occ_A and xP > xA
    vis_B = fov_B_ok and not occ_B and xP > xB

    if vis_A and det_time_A is None:
        det_time_A = t
    if vis_B and det_time_B is None:
        det_time_B = t

# ─── TTC (Time-To-Collision) ──────────────────────────────────────────────
# Αν Α ανιχνεύσει πεζό στο det_time_A, χρόνος TTC:
if det_time_A is not None:
    x_ped_at_det  = ped_x
    x_car_at_det  = x0_A + v_ego * det_time_A
    gap_at_det    = x_ped_at_det - x_car_at_det
    ttc_A         = gap_at_det / v_ego
else:
    ttc_A         = 0.0

if det_time_B is not None:
    x_car_B_det   = x0_A + v_ego * det_time_B  # Α λαμβάνει warning από Β
    gap_B         = ped_x - x_car_B_det
    ttc_B         = gap_B / v_ego
else:
    ttc_B         = 0.0

detection_advantage = (det_time_A or t_max) - (det_time_B or t_max)
ttc_advantage       = ttc_B - ttc_A

print("=" * 64)
print("ΙΔΕΑ 3 — Blind Spot Coverage (Cooperative)")
print("=" * 64)
print(f"Ταχύτητα οχημάτων:      {v_ego*3.6:.0f} km/h")
print(f"FOV αισθητήρα:          ±{fov_half_deg:.0f}°")
print(f"Χρόνος ανίχνευσης (Α):  {det_time_A:.2f} s" if det_time_A else "Α: Δεν ανιχνεύει")
print(f"Χρόνος ανίχνευσης (Β):  {det_time_B:.2f} s" if det_time_B else "Β: Δεν ανιχνεύει")
print(f"Πλεονέκτημα Β:          {detection_advantage:.2f} s νωρίτερα")
print(f"TTC αν μόνο Α:          {ttc_A:.2f} s")
print(f"TTC με warning από Β:   {ttc_B:.2f} s")
print(f"Επιπλέον χρόνος αντίδρ: {ttc_advantage:.2f} s")

# ─── Figure ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Ιδέα 3 — Blind Spot Coverage μέσω Cooperative Sensing\n"
             f"Αμάξι Β ανιχνεύει {detection_advantage:.1f}s νωρίτερα — "
             f"επιπλέον {ttc_advantage:.1f}s για αντίδραση",
             fontsize=11, fontweight="bold")

# Plot 1: Σκηνή τη στιγμή t_show
t_show = det_time_B + 0.3 if det_time_B else 1.0
i_show = int(t_show / dt_sim)
i_show = min(i_show, len(time_steps)-1)

ax = axes[0]
ax.set_xlim(-5, 55)
ax.set_ylim(-6, 8)
ax.set_aspect("equal")
ax.set_facecolor(var:="var(--color-background-secondary)" if False else "#f8f8f8")

# Δρόμος
for lane in [0, lane_width, -lane_width]:
    ax.axhline(lane, color="gray", lw=0.5, ls="--", alpha=0.5)
ax.axhline(0, color="white", lw=2)
ax.fill_between([-5, 55], [-lane_width, -lane_width], [2*lane_width, 2*lane_width],
                alpha=0.08, color="gray")

# Truck
truck_patch = patches.Rectangle((truck_x, ped_y - truck_width/2),
                                  truck_length, truck_width,
                                  linewidth=1, edgecolor="#333", facecolor="#aaa", zorder=3)
ax.add_patch(truck_patch)
ax.text(truck_x + truck_length/2, ped_y, "TRUCK", ha="center", va="center", fontsize=8, fontweight="bold")

# Πεζός
px, py = ped_positions[i_show]
ax.plot(px, py, "o", ms=10, color="#E24B4A", zorder=5)
ax.annotate("Πεζός\n(κρυμμένος\nαπό Α)", (px, py), xytext=(px+2, py-3),
            fontsize=8, color="#A32D2D", arrowprops=dict(arrowstyle="->", color="#A32D2D"))

# Αμάξια
xA, yA = car_A_positions[i_show]
xB, yB = car_B_positions[i_show]
carA = patches.FancyBboxPatch((xA-1.5, yA-0.8), 3, 1.6, boxstyle="round,pad=0.1",
                               facecolor="#185FA5", edgecolor="#0C447C", zorder=4)
carB = patches.FancyBboxPatch((xB-1.5, yB-0.8), 3, 1.6, boxstyle="round,pad=0.1",
                               facecolor="#1D9E75", edgecolor="#0F6E56", zorder=4)
ax.add_patch(carA)
ax.add_patch(carB)
ax.text(xA, yA, "Α", ha="center", va="center", color="white", fontweight="bold", fontsize=10, zorder=5)
ax.text(xB, yB, "Β", ha="center", va="center", color="white", fontweight="bold", fontsize=10, zorder=5)

# FOV cones
for (cx, cy, col, label) in [(xA, yA, "#185FA5", "FOV Α"), (xB, yB, "#1D9E75", "FOV Β")]:
    cone = patches.Wedge((cx, cy), 25, -fov_half_deg, fov_half_deg,
                          alpha=0.08, color=col)
    ax.add_patch(cone)

# Occlusion shadow
shadow = patches.Polygon([(truck_x + truck_length, ped_y - truck_width/2),
                           (truck_x + truck_length + 20, ped_y - truck_width/2 - 5),
                           (truck_x + truck_length + 20, ped_y + truck_width/2),
                           (truck_x + truck_length, ped_y + truck_width/2)],
                          alpha=0.15, color="red", zorder=2)
ax.add_patch(shadow)
ax.text(truck_x + truck_length + 5, ped_y - 3.5, "Τυφλή\nζώνη Α",
        ha="center", fontsize=8, color="#A32D2D")

# V2X arrow
ax.annotate("", xy=(xA+2, yA+0.3), xytext=(xB+2, yB-0.3),
            arrowprops=dict(arrowstyle="->", color="#BA7517", lw=2))
ax.text((xA+xB)/2 + 3, (yA+yB)/2, "V2X\nwarning", fontsize=8, color="#633806", ha="center")

ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title(f"Σκηνή t = {t_show:.1f}s")
ax.grid(alpha=0.2)

# Plot 2: Detection timeline
ax = axes[1]
t_arr = time_steps

# Visibility over time
vis_A_arr = []
vis_B_arr = []
for i, t in enumerate(t_arr):
    xA2, yA2 = car_A_positions[i]
    xB2, yB2 = car_B_positions[i]
    xP2, yP2 = ped_positions[i]

    fA = in_fov((xA2, yA2), (xP2, yP2), 0, fov_half)
    fB = in_fov((xB2, yB2), (xP2, yP2), 0, fov_half)
    oA = is_occluded((xA2, yA2), (xP2, yP2), obstacle) if fA else True
    oB = is_occluded((xB2, yB2), (xP2, yP2), obstacle) if fB else True
    vis_A_arr.append(1 if (fA and not oA and xP2 > xA2) else 0)
    vis_B_arr.append(1 if (fB and not oB and xP2 > xB2) else 0)

ax.fill_between(t_arr, vis_B_arr, alpha=0.3, color="#1D9E75", label="Β ανιχνεύει")
ax.fill_between(t_arr, vis_A_arr, alpha=0.3, color="#185FA5", label="Α ανιχνεύει")
ax.step(t_arr, vis_B_arr, where="post", color="#1D9E75", lw=2)
ax.step(t_arr, vis_A_arr, where="post", color="#185FA5", lw=2)

if det_time_B: ax.axvline(det_time_B, color="#1D9E75", ls="--", lw=1.5, label=f"Β detection @ {det_time_B:.1f}s")
if det_time_A: ax.axvline(det_time_A, color="#185FA5", ls="--", lw=1.5, label=f"Α detection @ {det_time_A:.1f}s")

ax.set_xlabel("Χρόνος (s)")
ax.set_ylabel("Ορατότητα πεζού")
ax.set_title(f"Timeline ανίχνευσης — Β κερδίζει {detection_advantage:.1f}s")
ax.set_yticks([0, 1])
ax.set_yticklabels(["Αόρατος", "Ορατός"])
ax.set_ylim(-0.1, 1.4)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("/home/claude/iscai_extensions/idea03_blind_spot_coverage.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: idea03_blind_spot_coverage.png")
