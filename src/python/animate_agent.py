"""
animate_agent.py — 8B: Animated GIF replay
Loads agent_trajectory.npy and renders a colour-coded GIF of the
agent sweeping through the arena, with food dots and a heading arrow.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import math

data = np.load("agent_trajectory.npy", allow_pickle=True).item()
xs = data["x"]
ys = data["y"]
hs = data.get("heading", [0.0] * len(xs))
atps = data["atp"]
das = data["da"]
N = len(xs)

ARENA_W, ARENA_H = 800, 600
AGENT_R = 15
TRAIL = 20  # how many past positions to draw as fading trail

fig, axes = plt.subplots(
    1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [3, 1]}, facecolor="#0d0d0d"
)
ax_arena, ax_tele = axes
fig.suptitle("Leviathan v2.0 — Live Agent Replay", color="white", fontsize=13)

ax_arena.set_facecolor("#111122")
ax_arena.set_xlim(0, ARENA_W)
ax_arena.set_ylim(0, ARENA_H)
ax_arena.set_aspect("equal")
ax_arena.set_title("2D Arena", color="white")
ax_arena.tick_params(colors="gray")
for sp in ax_arena.spines.values():
    sp.set_edgecolor("#333")

ax_tele.set_facecolor("#0a0a14")
ax_tele.set_xlim(0, N)
ax_tele.set_ylim(0, max(max(atps), 1) * 1.1)
ax_tele.set_title("ATP vs DA", color="white")
ax_tele.tick_params(colors="gray")
ax_tele_r = ax_tele.twinx()
for sp in ax_tele.spines.values():
    sp.set_edgecolor("#333")
ax_tele_r.tick_params(colors="gray")

# Static telemetry lines
ax_tele.plot(range(N), atps, color="#00e5ff", lw=1.2, alpha=0.4, label="ATP")
ax_tele_r.plot(
    range(N), das, color="#ff6f00", lw=1.2, alpha=0.4, linestyle="--", label="DA"
)
ax_tele.set_ylabel("ATP", color="#00e5ff", fontsize=8)
ax_tele_r.set_ylabel("DA", color="#ff6f00", fontsize=8)

# Animated elements
(trail_lines,) = ax_arena.plot([], [], color="#6060ff", lw=1.5, alpha=0.6)
agent_circle = plt.Circle((xs[0], ys[0]), AGENT_R, color="#00e5ff", zorder=5)
ax_arena.add_patch(agent_circle)

arrow_len = AGENT_R * 1.8
arrow = ax_arena.annotate(
    "",
    xy=(xs[0], ys[0]),
    xytext=(xs[0], ys[0]),
    arrowprops=dict(arrowstyle="->", color="yellow", lw=1.5),
)

tick_line = ax_tele.axvline(0, color="white", lw=0.8, alpha=0.6)
tick_line_r = ax_tele_r.axvline(0, color="white", lw=0.8, alpha=0.6)

time_text = ax_arena.text(10, ARENA_H - 20, "", color="white", fontsize=8)


def init():
    trail_lines.set_data([], [])
    return trail_lines, agent_circle, arrow, tick_line, time_text


def update(frame):
    i = frame
    # Trail
    start = max(0, i - TRAIL)
    trail_lines.set_data(xs[start : i + 1], ys[start : i + 1])

    # Agent body
    agent_circle.center = (xs[i], ys[i])

    # Heading arrow
    h = hs[i]
    tip_x = xs[i] + arrow_len * math.cos(h)
    tip_y = ys[i] + arrow_len * math.sin(h)
    arrow.set_position((xs[i], ys[i]))
    arrow.xy = (tip_x, tip_y)

    # Telemetry cursor
    tick_line.set_xdata([i, i])
    tick_line_r.set_xdata([i, i])

    time_text.set_text(f"step {i*30}  ATP={atps[i]:.0f}  DA={das[i]:.2f}")
    return trail_lines, agent_circle, arrow, tick_line, time_text


ani = animation.FuncAnimation(
    fig, update, frames=N, init_func=init, interval=60, blit=False
)

writer = animation.PillowWriter(fps=15)
ani.save("replay.gif", writer=writer, dpi=90)
print("Saved replay.gif")
plt.close()
