"""
render_agent.py
Renders the saved Leviathan agent trajectory with telemetry charts.
Usage:
    source venv/bin/activate
    PYTHONPATH=src/python python3 src/python/render_agent.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math

data = np.load("agent_trajectory.npy", allow_pickle=True).item()

x = data["x"]
y = data["y"]
heading = data.get("heading", [0.0] * len(x))
atp = data["atp"]
da = data["da"]
na = data.get("na", [0.0] * len(atp))
t_l = data.get("torque_l", [0.0] * len(atp))
t_r = data.get("torque_r", [0.0] * len(atp))
ticks = list(range(len(atp)))

fig = plt.figure(figsize=(16, 9), facecolor="#0d0d0d")
fig.suptitle(
    "Leviathan v2.0 — Embodied Agent Simulation", color="white", fontsize=15, y=0.97
)

gs = gridspec.GridSpec(
    3,
    2,
    figure=fig,
    left=0.05,
    right=0.97,
    top=0.92,
    bottom=0.06,
    wspace=0.3,
    hspace=0.45,
)

# ── LEFT: 2D Arena Trajectory ──────────────────────────────────────────────
ax_arena = fig.add_subplot(gs[:, 0], facecolor="#111122")
ax_arena.set_xlim(0, 800)
ax_arena.set_ylim(0, 600)
ax_arena.set_aspect("equal")
ax_arena.set_title("2D Arena (800 × 600)", color="white")
ax_arena.tick_params(colors="gray")
for spine in ax_arena.spines.values():
    spine.set_edgecolor("#333")

# Colour gradient: blue -> red over time
n = len(x)
for i in range(n - 1):
    frac = i / max(n - 2, 1)
    r, g, b = frac, 0.3, 1.0 - frac
    ax_arena.plot(x[i : i + 2], y[i : i + 2], color=(r, g, b), lw=1.2, alpha=0.8)

# Heading arrows every 10 samples
ARROW_LEN = 18
for i in range(0, n, max(1, n // 30)):
    h = heading[i]
    ax_arena.annotate(
        "",
        xy=(x[i] + ARROW_LEN * math.cos(h), y[i] + ARROW_LEN * math.sin(h)),
        xytext=(x[i], y[i]),
        arrowprops=dict(arrowstyle="->", color="yellow", lw=0.8),
    )

# Start / End markers
ax_arena.scatter(x[0], y[0], color="lime", s=80, zorder=6, label="Start")
ax_arena.scatter(x[-1], y[-1], color="red", s=80, zorder=6, label="End")
ax_arena.legend(facecolor="#222", labelcolor="white", loc="upper right", fontsize=8)

# ── RIGHT TOP: ATP & DA timeline ───────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 1], facecolor="#0a0a14")
ax1.plot(ticks, atp, color="#00e5ff", lw=1.4, label="Mean ATP")
ax1.set_ylabel("ATP", color="#00e5ff", fontsize=8)
ax1.set_title("Energy & Dopamine", color="white", fontsize=9)
ax1.tick_params(colors="gray", labelsize=7)
ax1b = ax1.twinx()
ax1b.plot(ticks, da, color="#ff6f00", lw=1.4, linestyle="--", label="DA")
ax1b.set_ylabel("DA", color="#ff6f00", fontsize=8)
ax1b.tick_params(colors="gray", labelsize=7)
for spine in ax1.spines.values():
    spine.set_edgecolor("#333")
for spine in ax1b.spines.values():
    spine.set_edgecolor("#333")
lines1, lbl1 = ax1.get_legend_handles_labels()
lines2, lbl2 = ax1b.get_legend_handles_labels()
ax1.legend(
    lines1 + lines2,
    lbl1 + lbl2,
    facecolor="#222",
    labelcolor="white",
    fontsize=7,
    loc="upper right",
)

# ── RIGHT MID: Noradrenaline (arousal) ─────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 1], facecolor="#0a0a14")
ax2.plot(ticks, na, color="#b39ddb", lw=1.4, label="NA (Arousal)")
ax2.set_ylabel("NA", color="#b39ddb", fontsize=8)
ax2.set_title("Noradrenaline (Arousal / Stress)", color="white", fontsize=9)
ax2.tick_params(colors="gray", labelsize=7)
for spine in ax2.spines.values():
    spine.set_edgecolor("#333")
ax2.legend(facecolor="#222", labelcolor="white", fontsize=7)

# ── RIGHT BOTTOM: Left vs Right Torque ─────────────────────────────────────
ax3 = fig.add_subplot(gs[2, 1], facecolor="#0a0a14")
ax3.plot(ticks, t_l, color="#69f0ae", lw=1.2, label="Torque L")
ax3.plot(ticks, t_r, color="#ff5252", lw=1.2, label="Torque R", linestyle="--")
ax3.set_ylabel("Torque", color="white", fontsize=8)
ax3.set_xlabel("Sample step (each = 30 ticks)", color="gray", fontsize=7)
ax3.set_title("Cerebellum → Motor Torque (L vs R)", color="white", fontsize=9)
ax3.tick_params(colors="gray", labelsize=7)
for spine in ax3.spines.values():
    spine.set_edgecolor("#333")
ax3.legend(facecolor="#222", labelcolor="white", fontsize=7)

plt.savefig("trajectory.png", dpi=150, facecolor=fig.get_facecolor())
print("Saved trajectory.png")
plt.close()
