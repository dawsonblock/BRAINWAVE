# BRAINWAVE — Leviathan v2.0

> **A fully continuous, embodied dynamical system where intelligence emerges from the phase-synchronisation of non-linear oscillators bound by physical constraints.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.x%20ready-76b900?logo=nvidia)](https://developer.nvidia.com/cuda)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Architecture Overview

Leviathan abandons discrete, feed-forward computation. Each node is a **physical pendulum** — it has mass, friction, and communicates with other nodes after a realistic **axonal conduction delay**.

```
┌──────────────────────────────────────────────────────────┐
│                   LEVIATHAN v2.0 ENGINE                  │
│                                                          │
│  Sensory Input (Raycasts)                                │
│       │                                                  │
│       ▼                                                  │
│  ┌─ Thalamus ──────────┐    4-Channel Endocrine System   │
│  │  S_i(t) forcing     │◄── DA / NA / SER / ACH          │
│  └─────────────────────┘         ▲                       │
│       │                          │ ATP Economy            │
│       ▼                          │                       │
│  ┌─ Cortex ────────────┐    ┌─── Metabolism ──┐          │
│  │  DDE oscillators    │───►│  drain + prune  │          │
│  │  PTDP plasticity    │    └─────────────────┘          │
│  └─────────────────────┘                                 │
│       │                                                  │
│       ▼                                                  │
│  ┌─ Cerebellum ────────┐                                 │
│  │  Motor Decoder      │──► Left / Right Torque          │
│  └─────────────────────┘         │                       │
│                                  ▼                       │
│                         2D Embodied Arena                 │
│                    (food, walls, reward loop)             │
└──────────────────────────────────────────────────────────┘
```

---

## Core Mathematics

### Second-Order Delay Differential Equation (DDE)

Each node $i$ evolves as a physical pendulum:

$$m_i \ddot{\phi}_i + c_i (\dot{\phi}_i - \omega_i) = \text{SER} \sum_j W_{ij} \sin(\phi_j(t-\tau_{ij}) - \phi_i) + S_i(t) + \text{NA} \cdot \xi$$

| Variable | Meaning |
|---|---|
| $m_i$ | Membrane inertia (modulated by ACH) |
| $c_i$ | Damping coefficient |
| $\omega_i$ | Intrinsic frequency |
| $\tau_{ij}$ | Axonal conduction delay ($d_{ij} / v_c$) |
| $S_i(t)$ | Sensory forcing |
| $\text{NA} \cdot \xi$ | Noradrenergic noise |

### Phase-Timing-Dependent Plasticity (PTDP)

$$\Delta W_{ij} = \begin{cases} \text{DA} \cdot A_+ e^{-\Delta\phi / \tau_+} & \text{if } \Delta\phi > 0 \text{ (j leads i)} \\ -A_- e^{\Delta\phi / \tau_-} & \text{otherwise} \end{cases}$$

### ATP Metabolic Economy

$$\frac{dE_i}{dt} = -\left(R_\text{basal} + k_\text{cost} |\dot{\phi}_i| + k_\text{maint} \sum_j W_{ij}\right)$$

Food consumption injects **+500 ATP** and spikes **Dopamine** to reinforce the neural pathway.

---

## Project Structure

```
BRAINWAVE/
├── docs/
│   └── leviathan_spec.md          # Full engineering specification
├── src/
│   ├── python/
│   │   ├── leviathan/             # Core PyTorch brain
│   │   │   ├── network.py         # LeviathanNetwork (DDE engine)
│   │   │   ├── topology.py        # 3D spatial init (Thalamus/Cortex/Cerebellum)
│   │   │   ├── endocrine.py       # ATP metabolism + neuromodulator dynamics
│   │   │   ├── motor.py           # Cerebellum → torque decoder
│   │   │   ├── plasticity.py      # PTDP weight updates
│   │   │   └── config.py          # All physical constants
│   │   ├── environment/
│   │   │   └── arena.py           # 2D embodied arena (walls, food, raycasts)
│   │   ├── experiments/
│   │   │   └── ptdp_ablation.py   # Learning verification study
│   │   ├── run_agent.py           # Embodied simulation entry point
│   │   ├── animate_agent.py       # GIF replay generator
│   │   └── render_agent.py        # Static trajectory + telemetry plot
│   ├── cuda/
│   │   ├── kernels.h              # GPU state struct + kernel declarations
│   │   ├── kernels.cu             # CUDA kernel implementations
│   │   └── main.cu                # Host memory allocation + kernel loop
│   ├── server/
│   │   ├── server.py              # FastAPI WebSocket live dashboard
│   │   └── static/index.html      # Canvas real-time visualiser
│   └── visualization/
│       └── viewer.py              # Plotly 3D network topology viewer
└── CMakeLists.txt                 # CUDA build system
```

---

## Quick Start

### 1. Python Prototype (works on any machine)

```bash
git clone https://github.com/dawsonblock/BRAINWAVE.git
cd BRAINWAVE
python3 -m venv venv && source venv/bin/activate
pip install torch numpy plotly matplotlib scipy

# Run the embodied agent simulation (3000 ticks)
PYTHONPATH=src/python python3 src/python/run_agent.py

# Generate the static telemetry plot
PYTHONPATH=src/python python3 src/python/render_agent.py

# Generate an animated GIF replay
PYTHONPATH=src/python python3 src/python/animate_agent.py
```

### 2. Real-Time Web Dashboard

```bash
pip install fastapi uvicorn websockets
PYTHONPATH=src/python uvicorn src.server.server:app --port 8765
# Open http://localhost:8765
```

### 3. PTDP Learning Verification

```bash
PYTHONPATH=src/python python3 src/python/experiments/ptdp_ablation.py
# Generates ptdp_ablation.png
```

### 4. CUDA / C++ Build (requires NVIDIA GPU + CUDA 12)

```bash
mkdir build && cd build
cmake -DCMAKE_CUDA_ARCHITECTURES=native ..
make -j4
./leviathan_sim
```

---

## Phase Roadmap

| Phase | Status | Description |
|---|---|---|
| 1 — Formalize Math & Docs    | ✅ | DDE, PTDP, ATP equations documented |
| 2 — Python/PyTorch Prototype | ✅ | Full DDE integration engine |
| 3 — 3D Visualization         | ✅ | Plotly interactive topology viewer |
| 4 — CUDA Foundation          | ✅ | CSR sparse matrices + ring buffers |
| 5 — Neuromodulator Dynamics  | ✅ | Dynamic DA/NA/SER/ACH equations |
| 6 — Core CUDA Kernels        | ✅ | All 4 mathematical kernels written |
| 7 — Embodied Environment     | ✅ | 2D arena, sensor raycasts, motor torque |
| 8A — PTDP Ablation           | ✅ | Learning curve comparison |
| 8B — Animated Replay         | ✅ | .gif trajectory animation |
| 8C — README                  | ✅ | This file |
| 8D — Obstacle Avoidance      | ✅ | Wall raycasts + agent collision |
| 8E — Web Dashboard           | ✅ | FastAPI WebSocket real-time canvas |

---

## License

MIT © 2026 Dawson Block
