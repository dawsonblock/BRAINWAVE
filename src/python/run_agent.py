import torch
import numpy as np
from environment.arena import Arena
from leviathan.network import LeviathanNetwork
from leviathan.motor import MotorDecoder
from leviathan.endocrine import update_metabolism, update_neuromodulators
from leviathan.config import (
    DA_REWARD_SPIKE,
    ATP_FOOD_REWARD,
    SENSORY_GAIN,
    MOTOR_GAIN,
    WALL_GAIN,
)


def run_agent_simulation(ticks=3000):
    print("Initializing Leviathan v2.0 Agent Simulation...")

    # 1. Setup Brain
    num_nodes = 200
    brain = LeviathanNetwork(num_nodes)

    # 5 Raycast Sensors -> Thalamus nodes 0-4
    sensor_indices = list(range(5))

    # MotorDecoder reads Cerebellum region labels from the brain
    motor_decoder = MotorDecoder(network=brain)

    # 2. Setup Body/Environment
    arena = Arena(width=800, height=600, num_foods=20)
    arena.add_default_obstacles()  # 8D: add interior wall obstacles

    # Logging
    history = {
        "x": [],
        "y": [],
        "heading": [],
        "atp": [],
        "da": [],
        "na": [],
        "torque_l": [],
        "torque_r": [],
        "foods_eaten": 0,
    }

    print(f"Starting simulation for {ticks} ticks...")
    for tick in range(ticks):

        # ---- SENSORY PHASE ----
        # Dual-channel raycasts: food proximity + wall danger
        food_prox, wall_prox = arena.get_sensory_raycasts(
            num_rays=5, fov_deg=120, max_dist=250.0
        )

        # Thalamus nodes 0-4: food proximity (approach bias)
        # Thalamus nodes 5-9: wall proximity (avoidance signal)
        s_input = torch.zeros(num_nodes)
        for i, prox in enumerate(food_prox):
            s_input[sensor_indices[i]] = SENSORY_GAIN * float(prox)
        wall_indices = list(range(5, 10))
        for i, prox in enumerate(wall_prox):
            if wall_indices[i] < num_nodes:
                s_input[wall_indices[i]] = WALL_GAIN * float(prox)

        # ---- COGNITIVE PHASE ----
        delayed_phi = brain.step(external_sensory_input=s_input)

        # ── Phase 12 Plasticity: PTDP + Active Inference Learning ─────
        if brain.current_tick > 10:
            from leviathan.plasticity import apply_ptdp, apply_active_inf_learning

            apply_ptdp(brain, delayed_phi)
            apply_active_inf_learning(brain, delayed_phi)

        update_metabolism(brain)
        update_neuromodulators(brain)

        # Export topology once for CUDA parity verification
        if tick == 0:
            from leviathan.topology import save_topology_bin

            # Fetch all required components for save_topology_bin
            save_topology_bin(
                "leviathan_topology_p12.bin",
                (
                    brain.row_ptr,
                    brain.col_idx,
                    brain.weights,
                    brain.delay_ticks,
                    brain.is_inhibitory,
                    brain.positions,
                    brain.region_labels,
                ),
            )

        # ---- MOTOR PHASE ----
        raw_l, raw_r = motor_decoder.decode_torque()

        # Adaptive motor gain: scale so that a typical phi_dot sum (~5-20 rad/s)
        # maps to a useful arena displacement per tick.
        # We also add a small forward bias so the agent explores even early on.
        torque_l = raw_l * MOTOR_GAIN + 2.0
        torque_r = raw_r * MOTOR_GAIN + 2.0

        foods_eaten = arena.move_agent(torque_l, torque_r)

        # ---- REWARD PHASE ----
        if foods_eaten > 0:
            history["foods_eaten"] += foods_eaten
            brain.atp += ATP_FOOD_REWARD * foods_eaten
            brain.DA = min(brain.DA + DA_REWARD_SPIKE * foods_eaten, 10.0)
            print(
                f"  [Tick {tick:>5}] Ate food! ATP={brain.atp.mean().item():.1f}  "
                f"DA={brain.DA:.2f}  pos=({arena.agent_x:.0f},{arena.agent_y:.0f})"
            )

        # ---- LOGGING (every 30 ticks) ----
        if tick % 30 == 0:
            history["x"].append(arena.agent_x)
            history["y"].append(arena.agent_y)
            history["heading"].append(arena.agent_heading)
            history["atp"].append(brain.atp.mean().item())
            history["da"].append(brain.DA)
            history["na"].append(brain.NA)
            history["torque_l"].append(torque_l)
            history["torque_r"].append(torque_r)

    print("\nSimulation Complete.")
    print(f"  Total Food Eaten : {history['foods_eaten']}")
    print(f"  Final Avg ATP    : {history['atp'][-1]:.1f}")
    print(f"  Final DA         : {history['da'][-1]:.3f}")

    np.save("agent_trajectory.npy", history)
    print("Saved agent trajectory to agent_trajectory.npy")


if __name__ == "__main__":
    run_agent_simulation()
