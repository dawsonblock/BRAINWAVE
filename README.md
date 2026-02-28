<div align="center">

# ⚡ BRAINWAVE — Leviathan v2.0

**A continuous, embodied, self-regulating AGI architecture  
built on non-linear oscillators, axonal delays, and ATP-driven survival.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776ab?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.x%20Ready-76b900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda)
[![FastAPI](https://img.shields.io/badge/FastAPI-WebSocket%20Dashboard-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-22c55e)](LICENSE)

</div>

---

## What Is Leviathan?

Most neural networks are static graphs of matrix multiplications. Leviathan is different.

Every node is a **physical pendulum** — it has mass, damping, and a natural frequency. Nodes communicate through **spike-like phase signals** that travel along simulated axons at finite speed, introducing realistic conduction delays. The system never "forward passes." It runs continuously, like a brain.

The network is housed in a **2D physical body** with eyes (raycasts) and motors (tank-drive). Its only drive is survival: find food to replenish ATP or its synapses will be pruned and it will starve. Intelligence emerges from this pressure.

---

## Architecture

```
  ┌───────────────────────────────────────────────────────────────────┐
  │                      LEVIATHAN v2.0 ENGINE                        │
  │                                                                   │
  │  World ──► 5× Raycast Sensors ──► Thalamus [S_i(t) forcing]      │
  │          + 5× Wall Danger Rays ──► Thalamus [nodes 5–9]          │
  │                         │                                         │
  │                         ▼                                         │
  │              ┌─────────────────────┐                              │
  │              │    Cortex (DDEs)    │◄──── Endocrine System        │
  │              │  200 oscillators    │      DA / NA / SER / ACH     │
  │              │  PTDP plasticity    │◄──── ATP Economy             │
  │              └─────────────────────┘      (drain → prune → die)   │
  │                         │                                         │
  │                         ▼                                         │
  │              ┌─────────────────────┐                              │
  │              │ Cerebellum (Motor)  │──► Left / Right Torque       │
  │              └─────────────────────┘          │                   │
  │                                               ▼                   │
  │                              Tank-Drive Body in 2D Arena          │
  │                              (food, walls, ATP reward loop)       │
  └───────────────────────────────────────────────────────────────────┘
```

---

## Core Mathematics

### Delay Differential Equation (DDE) — Second-Order Pendulum

$$m_i \ddot{\phi}_i + c_i(\dot{\phi}_i - \omega_i) = \underbrace{\text{SER} \sum_j W_{ij} \sin\!\bigl(\phi_j(t - \tau_{ij}) - \phi_i\bigr)}_{\text{synaptic coupling}} + \underbrace{S_i(t)}_{\text{sensory}} + \underbrace{\text{NA}\cdot\xi}_{\text{noise}}$$

| Symbol | Meaning |
|--------|---------|
| $m_i / \text{ACH}$ | Inertia (modulated by Acetylcholine) |
| $\tau_{ij} = d_{ij}/v_c$ | Axonal conduction delay |
| $\text{SER}$ | Serotonin scales global coupling strength |
| $S_i(t)$ | Raycast-derived sensory forcing |

### Phase-Timing-Dependent Plasticity (PTDP)

$$\Delta W_{ij} = \begin{cases} \text{DA} \cdot A_+ \, e^{-\Delta\phi/\tau_+} & \Delta\phi > 0 \;\; \text{(j leads i, reinforce)} \\ -A_- \, e^{\;\Delta\phi/\tau_-} & \Delta\phi \leq 0 \;\; \text{(i leads j, weaken)} \end{cases}$$

Reward (eating food) spikes **Dopamine → DA**, which gates the LTP window wider — the exact pathway that worked is strengthened.

### ATP Metabolic Economy

$$\frac{dE_i}{dt} = -\!\left(R_\text{basal} + k_\text{cost}|\dot{\phi}_i| + k_\text{maint}\sum_j W_{ij}\right)$$

- Starvation ($E_i \to 0$): weakest incoming synapse is pruned.
- Food collision: **+500 ATP** injected, **DA** spiked.

---

## PTDP Ablation — Does Plasticity Help?

Running two identical agents for 5,000 ticks (same random seed):

| Condition | Food Eaten |
|-----------|-----------|
| PTDP **ON** | **500** |
| PTDP **OFF** | 352 |
| **Improvement** | **+42%** |

Synaptic plasticity demonstrably increases survival. The brain learns the motor patterns that led to food.

---

## Project Structure

```
BRAINWAVE/
├── README.md
├── docs/leviathan_spec.md          ← Full engineering specification
└── src/
    ├── python/
    │   ├── leviathan/
    │   │   ├── network.py          ← DDE oscillator engine
    │   │   ├── topology.py         ← 3-region spatial init (Small-World)
    │   │   ├── endocrine.py        ← ATP + neuromodulator dynamics
    │   │   ├── motor.py            ← Cerebellum → torque decoder
    │   │   ├── plasticity.py       ← PTDP weight updates
    │   │   └── config.py           ← Physical constants
    │   ├── environment/
    │   │   └── arena.py            ← 2D arena: food, walls, raycasts
    │   ├── experiments/
    │   │   └── ptdp_ablation.py    ← Learning curve comparison
    │   ├── run_agent.py            ← Embodied simulation entry point
    │   ├── animate_agent.py        ← Animated GIF replay
    │   └── render_agent.py         ← Static trajectory + telemetry plot
    ├── cuda/
    │   ├── kernels.h / kernels.cu  ← GPU kernel implementations
    │   └── main.cu                 ← Host memory + simulation loop
    ├── server/
    │   ├── server.py               ← FastAPI WebSocket live server
    │   └── static/index.html       ← Canvas real-time dashboard
    └── visualization/
        └── viewer.py               ← Plotly 3D topology viewer
```

---

## Quick Start

### Run the Embodied Agent

```bash
git clone https://github.com/dawsonblock/BRAINWAVE.git && cd BRAINWAVE
python3 -m venv venv && source venv/bin/activate
pip install torch numpy matplotlib scipy

# 3000-tick simulation with wall obstacles (logs food eaten to stdout)
PYTHONPATH=src/python python3 src/python/run_agent.py

# Static trajectory + telemetry chart → trajectory.png
PYTHONPATH=src/python python3 src/python/render_agent.py

# Animated GIF replay → replay.gif
PYTHONPATH=src/python python3 src/python/animate_agent.py
```

### Real-Time Web Dashboard

```bash
pip install fastapi uvicorn websockets
PYTHONPATH=src/python uvicorn src.server.server:app --port 8765
# Open http://localhost:8765
```

The dashboard streams the live arena (agent, food, walls) and animated telemetry bars for all 4 neuromodulators at ~20fps using a single WebSocket.

### PTDP Learning Verification

```bash
PYTHONPATH=src/python python3 src/python/experiments/ptdp_ablation.py
# Saves ptdp_ablation.png  (PTDP ON vs OFF survival curves)
```

### CUDA Build (requires NVIDIA GPU + CUDA 12+)

```bash
mkdir build && cd build
cmake -DCMAKE_CUDA_ARCHITECTURES=native ..
make -j4 && ./leviathan_sim
```

---

## Neuromodulator Dynamics

| Hormone | Role | Drive Signal |
|---------|------|-------------|
| **DA** (Dopamine) | Reward → gates PTDP | ATP above target + food reward |
| **NA** (Noradrenaline) | Arousal / stress | Starvation rate |
| **SER** (Serotonin) | Coupling strength multiplier | ATP health |
| **ACH** (Acetylcholine) | Inertia reduction (attention) | Phase velocity novelty |

All four decay back to baseline between events following $\dot{x} = -(x - x_0)/\tau + \alpha \cdot \text{drive}$.

---

## Phase Roadmap

| # | Phase | Status |
|---|-------|--------|
| 1 | Formalized Math & Docs | ✅ |
| 2 | Python / PyTorch DDE Engine | ✅ |
| 3 | 3D Topology Visualizer | ✅ |
| 4 | CUDA / C++ Foundation | ✅ |
| 5 | Dynamic Neuromodulators & Git Deploy | ✅ |
| 6 | Core CUDA Kernels | ✅ |
| 7 | 2D Embodied Arena | ✅ |
| 8A | PTDP Ablation Study | ✅ |
| 8B | Animated GIF Replay | ✅ |
| 8C | README | ✅ |
| 8D | Obstacle Avoidance (wall raycasts) | ✅ |
| 8E | Real-Time Web Dashboard | ✅ |

---

## License

MIT © 2026 Dawson Block
