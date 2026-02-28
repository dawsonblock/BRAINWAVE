"""
server.py — 8E: Real-Time Web Dashboard
FastAPI + WebSocket server that runs the Leviathan embodied agent and
streams live simulation state to connected browsers at ~20fps.

Usage:
    cd /Users/dawsonblock/Downloads/BRAINWAVE
    source venv/bin/activate
    pip install fastapi uvicorn websockets
    PYTHONPATH=src/python uvicorn src.server.server:app --reload --port 8765
Then open: http://localhost:8765
"""

import asyncio
import json
import sys
import os

# Make sure leviathan package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from environment.arena import Arena
from leviathan.network import LeviathanNetwork
from leviathan.motor import MotorDecoder
from leviathan.endocrine import update_metabolism, update_neuromodulators
from leviathan.config import DA_REWARD_SPIKE, ATP_FOOD_REWARD

SENSORY_GAIN = 80.0
MOTOR_GAIN = 10.0
TICK_SLEEP = 0.05  # 20 fps simulation rate

app = FastAPI(title="Leviathan v2.0 — Live Dashboard")

# ── Serve static files (the canvas frontend) ────────────────────────────
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    with open(os.path.join(static_dir, "index.html"), "r") as f:
        return f.read()


# ── Shared sim state (one global brain for simplicity) ──────────────────
class SimState:
    def __init__(self):
        self.brain = LeviathanNetwork(200)
        self.arena = Arena(width=800, height=600, num_foods=20)
        self.arena.add_default_obstacles()
        self.motor_dec = MotorDecoder(network=self.brain)
        self.sensor_idx = list(range(5))
        self.wall_idx = list(range(5, 10))
        self.foods_eaten = 0
        self.tick = 0

    def step(self):
        brain = self.brain
        arena = self.arena

        food_prox, wall_prox = arena.get_sensory_raycasts(
            num_rays=5, fov_deg=120, max_dist=250.0
        )

        s_input = torch.zeros(brain.num_nodes)
        for i, p in enumerate(food_prox):
            s_input[self.sensor_idx[i]] = SENSORY_GAIN * float(p)
        for i, p in enumerate(wall_prox):
            # Wall danger drives Noradrenaline sensing nodes (5-9)
            idx = self.wall_idx[i]
            if idx < brain.num_nodes:
                s_input[idx] = SENSORY_GAIN * 0.6 * float(p)

        brain.step(external_sensory_input=s_input)
        update_metabolism(brain)
        update_neuromodulators(brain)

        raw_l, raw_r = self.motor_dec.decode_torque()
        torque_l = raw_l * MOTOR_GAIN + 2.0
        torque_r = raw_r * MOTOR_GAIN + 2.0
        eaten = arena.move_agent(torque_l, torque_r)

        if eaten > 0:
            self.foods_eaten += eaten
            brain.atp += ATP_FOOD_REWARD * eaten
            brain.DA = min(brain.DA + DA_REWARD_SPIKE * eaten, 10.0)

        self.tick += 1
        return arena.to_dict(brain) | {
            "tick": self.tick,
            "total_food": self.foods_eaten,
        }


sim = SimState()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            state = sim.step()
            await websocket.send_text(json.dumps(state))
            await asyncio.sleep(TICK_SLEEP)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WS error: {e}")
