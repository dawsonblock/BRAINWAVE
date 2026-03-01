import torch
import torch.nn as nn
import math
import numpy as np
from environment.arena import Arena
from leviathan.network import LeviathanNetwork
from leviathan.motor import MotorDecoder
from leviathan.endocrine import update_metabolism, update_neuromodulators
from leviathan.evolution import Genome, EvoController
from leviathan.config import (
    DA_REWARD_SPIKE,
    ATP_FOOD_REWARD,
    SENSORY_GAIN,
    MOTOR_GAIN,
    WALL_GAIN,
)


def run_agent_simulation(ticks=1500, num_agents=4, num_generations=5):
    print(
        f"Initializing Leviathan Evolutionary Simulation (G={num_generations}, N={num_agents})..."
    )

    # 1. Setup Evolution
    evo = EvoController(population_size=num_agents)
    current_genomes = [Genome() for _ in range(num_agents)]

    for gen in range(num_generations):
        print(f"\n--- GENERATION {gen} ---")

        # Initialize Brains with Genomes
        num_nodes = 200
        brains = [LeviathanNetwork(num_nodes, genome=g) for g in current_genomes]
        motor_decoders = [MotorDecoder(network=b) for b in brains]

        # Sensor mappings
        sensor_indices = list(range(5))
        wall_indices = list(range(5, 10))
        peer_indices = list(range(15, 20))

        # Arena
        arena = Arena(width=800, height=600, num_foods=25, num_agents=num_agents)
        arena.add_default_obstacles()

        # Logging
        history = {
            "foods_eaten": 0,
            "last_torques": [(0.0, 0.0) for _ in range(num_agents)],
            "replay_buffer": [],
        }

        print(f"Starting GEN {gen} for {ticks} ticks...")
        for tick in range(ticks):
            for a_idx in range(num_agents):
                brain = brains[a_idx]
                decoder = motor_decoders[a_idx]

            # ---- SENSORY PHASE ----
            # Tri-channel raycasts: food, walls, peers
            food_prox, wall_prox, peer_prox = arena.get_sensory_raycasts(
                agent_idx=a_idx, num_rays=5, fov_deg=120, max_dist=250.0
            )

            s_input = torch.zeros(num_nodes)
            for i, prox in enumerate(food_prox):
                s_input[sensor_indices[i]] = SENSORY_GAIN * float(prox)
            for i, prox in enumerate(wall_prox):
                s_input[wall_indices[i]] = WALL_GAIN * float(prox)
            for i, prox in enumerate(peer_prox):
                s_input[peer_indices[i]] = 20.0 * float(prox)  # Peer avoidance gain

            # ---- PHASE 16/17: Chemical & Pheromone Sensing ----
            food_scent, r_phero, d_phero = arena.get_chemical_gradient(a_idx)

            # Food Scent: Limbic nodes 10-14
            for i in range(10, 15):
                s_input[i] += food_scent * 50.0

            # Social Pheromones: Modulate Endocrine directly (Social DA/ACH)
            # Reward pheromone from peers boost DA (Social Validation)
            # Danger pheromone from peers boost ACH (Social Alert)
            brain.DA = min(brain.DA + r_phero * 0.2, 10.0)
            brain.ACH = min(brain.ACH + d_phero * 0.5, 10.0)

            # ---- PHASE 16: LSM Memory Drive ----
            if brain.atp.mean() < 800.0:
                memory_vec = brain.lsm.readout(brain.phi_dot)
                association_indices = list(range(20, 40))
                drive_mag = 10.0 * (1.0 - (brain.atp.mean() / 800.0))
                for i in association_indices:
                    s_input[i] += drive_mag * torch.mean(torch.abs(memory_vec))

            # ---- PHASE 17: Neural Mimicry (Action Observation) ----
            # If peers are visible, inject their averaged torque into observer's brain
            # Observation region: indices 40-59
            if peer_prox.max() > 0.1:
                obs_indices = list(range(40, 60))
                # Simple proxy: average torque of all OTHER agents
                avg_l = sum(
                    t[0] for i, t in enumerate(history["last_torques"]) if i != a_idx
                ) / (num_agents - 1)
                avg_r = sum(
                    t[1] for i, t in enumerate(history["last_torques"]) if i != a_idx
                ) / (num_agents - 1)
                for i in obs_indices:
                    s_input[i] += (
                        avg_l + avg_r
                    ) * 5.0  # Injection of social motor state
            delayed_phi = brain.step(external_sensory_input=s_input)

            # ---- PLASTICITY PHASE ----
            if brain.current_tick > 10:
                from leviathan.plasticity import (
                    apply_ptdp,
                    apply_active_inf_learning,
                    apply_synaptic_scaling,
                )

                apply_ptdp(brain, delayed_phi)
                apply_active_inf_learning(brain, delayed_phi)
                apply_synaptic_scaling(brain)
                brain.synaptic_birth()

                # Train LSM
                target_vec = torch.zeros(2)
                if len(arena.foods) > 0:
                    agent_p = np.array([arena.agents_x[a_idx], arena.agents_y[a_idx]])
                    food_ps = np.array([[f[0], f[1]] for f in arena.foods])
                    diffs = food_ps - agent_p
                    dist_sq = np.sum(diffs**2, axis=1)
                    nearest_idx = np.argmin(dist_sq)
                    target_vec = (
                        torch.tensor(diffs[nearest_idx], dtype=torch.float32) / 500.0
                    )
                    brain.lsm_loss = brain.lsm.train_step(brain.phi_dot, target_vec)

            update_metabolism(brain)
            update_neuromodulators(brain)

            # ---- MOTOR PHASE ----
            raw_l, raw_r = decoder.decode_torque()
            torque_l = raw_l * MOTOR_GAIN + 2.0
            torque_r = raw_r * MOTOR_GAIN + 2.0

            foods_eaten = arena.move_agent(a_idx, torque_l, torque_r)
        if tick % 100 == 0:
            # Stats for first agent (as a representative)
            b0 = brains[0]
            print(
                f"  [Tick {tick:5d}] Ate={history['foods_eaten']} Lyap={b0.lyapunov_exponent:.3f} "
                f"Ent={b0.cognitive_entropy:.3f} E={b0.weights.shape[0]} DA={b0.DA:.2f}"
            )

        # ---- LOGGING (every 30 ticks) ----
        if tick % 30 == 0:
            history["atp"].append(
                float(torch.stack([b.atp.mean() for b in brains]).mean())
            )
            history["da"].append(float(sum(b.DA for b in brains) / num_agents))
            history["entropy"].append(
                float(sum(b.cognitive_entropy for b in brains) / num_agents)
            )

    # ---- OFFLINE REPLAY PHASE ----
    if history["replay_buffer"]:
        print(
            f"\nStarting Offline Replay (N={len(history['replay_buffer'])} memories)..."
        )
        for i, memory in enumerate(history["replay_buffer"]):
            brain = brains[memory["brain_idx"]]
            state = memory["state"]
            target = memory["target"]
            if target is not None:
                loss = brain.lsm.train_step(state, target)
                if i % 20 == 0:
                    print(
                        f"  [Replay {i:>3}] Brain {memory['brain_idx']} LSM Loss: {loss:.4f}"
                    )

        # End of Generation: Selection
        fitness_scores = []
        for b in brains:
            # Fitness = Final ATP + bonus for food eaten (not directly tracked here, so using ATP)
            fitness_scores.append(float(b.atp.mean()))

        print(f"Gen {gen} Best Fitness: {max(fitness_scores):.1f}")
        current_genomes = evo.select_and_reproduce(fitness_scores, current_genomes)

    print("\nEvolutionary Run Complete.")


if __name__ == "__main__":
    run_agent_simulation()
