"""
ptdp_ablation.py  –  8A: Learning Verification
Runs two identical agents (same random seed) for 5000 ticks each:
  - Agent A: PTDP enabled (normal)
  - Agent B: PTDP disabled (weights frozen)
Logs food collected per 100-tick window and saves plot.
"""

import sys
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Reproducible
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

sys.path.insert(0, "src/python")
from environment.arena import Arena
from leviathan.network import LeviathanNetwork
from leviathan.motor import MotorDecoder
from leviathan.endocrine import update_metabolism, update_neuromodulators
from leviathan.config import DA_REWARD_SPIKE, ATP_FOOD_REWARD

SENSORY_GAIN = 80.0
MOTOR_GAIN = 10.0
TICKS = 5000
WINDOW = 100  # bucket size for food-rate measurement
NUM_FOODS = 20


def run_agent(ptdp_enabled: bool, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    brain = LeviathanNetwork(200)
    sensor_idx = list(range(5))
    motor_dec = MotorDecoder(network=brain)
    arena = Arena(width=800, height=600, num_foods=NUM_FOODS)

    food_per_window = []
    window_count = 0

    for tick in range(TICKS):
        food_prox, _wall_prox = arena.get_sensory_raycasts(
            num_rays=5, fov_deg=120, max_dist=250.0
        )
        s_input = torch.zeros(brain.num_nodes)
        for i, p in enumerate(food_prox):
            s_input[sensor_idx[i]] = SENSORY_GAIN * float(p)

        delayed = brain.step(external_sensory_input=s_input)

        if ptdp_enabled:
            from leviathan.plasticity import apply_ptdp

            apply_ptdp(brain.weights, brain.phi, delayed, brain.DA)

        update_metabolism(brain)
        update_neuromodulators(brain)

        raw_l, raw_r = motor_dec.decode_torque()
        torque_l = raw_l * MOTOR_GAIN + 2.0
        torque_r = raw_r * MOTOR_GAIN + 2.0
        eaten = arena.move_agent(torque_l, torque_r)

        if eaten > 0:
            brain.atp += ATP_FOOD_REWARD * eaten
            brain.DA = min(brain.DA + DA_REWARD_SPIKE * eaten, 10.0)

        window_count += eaten
        if (tick + 1) % WINDOW == 0:
            food_per_window.append(window_count)
            window_count = 0

    return food_per_window


print("Running PTDP-enabled agent…")
ptdp_on = run_agent(ptdp_enabled=True, seed=SEED)
print("Running PTDP-disabled agent…")
ptdp_off = run_agent(ptdp_enabled=False, seed=SEED)

windows = np.arange(1, len(ptdp_on) + 1) * WINDOW

fig = plt.figure(figsize=(12, 5), facecolor="#0d0d0d")
gs = gridspec.GridSpec(
    1, 2, figure=fig, wspace=0.35, left=0.07, right=0.97, top=0.88, bottom=0.12
)

# Raw per-window
ax1 = fig.add_subplot(gs[0], facecolor="#0a0a14")
ax1.plot(windows, ptdp_on, color="#00e5ff", lw=1.8, label="PTDP ON")
ax1.plot(windows, ptdp_off, color="#ff5252", lw=1.8, linestyle="--", label="PTDP OFF")
ax1.set_title("Food Eaten per 100-Tick Window", color="white")
ax1.set_xlabel("Tick", color="gray")
ax1.set_ylabel("Food count", color="gray")
ax1.tick_params(colors="gray")
ax1.legend(facecolor="#222", labelcolor="white")
for sp in ax1.spines.values():
    sp.set_edgecolor("#333")

# Cumulative
ax2 = fig.add_subplot(gs[1], facecolor="#0a0a14")
ax2.plot(windows, np.cumsum(ptdp_on), color="#00e5ff", lw=1.8, label="PTDP ON")
ax2.plot(
    windows,
    np.cumsum(ptdp_off),
    color="#ff5252",
    lw=1.8,
    linestyle="--",
    label="PTDP OFF",
)
ax2.set_title("Cumulative Food Eaten", color="white")
ax2.set_xlabel("Tick", color="gray")
ax2.set_ylabel("Total food", color="gray")
ax2.tick_params(colors="gray")
ax2.legend(facecolor="#222", labelcolor="white")
for sp in ax2.spines.values():
    sp.set_edgecolor("#333")

fig.suptitle(
    "8A — PTDP Ablation Study: Does Plasticity Help Survival?",
    color="white",
    fontsize=13,
)

plt.savefig("ptdp_ablation.png", dpi=140, facecolor=fig.get_facecolor())
print(f"Saved ptdp_ablation.png")
print(f"PTDP ON  total: {sum(ptdp_on)}")
print(f"PTDP OFF total: {sum(ptdp_off)}")
