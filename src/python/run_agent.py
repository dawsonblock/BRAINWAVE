import torch
import numpy as np
from environment.arena import Arena
from leviathan.network import LeviathanNetwork
from leviathan.motor import MotorDecoder
from leviathan.endocrine import update_metabolism, update_neuromodulators
from leviathan.config import DA_REWARD_SPIKE, ATP_FOOD_REWARD


def run_agent_simulation(ticks=1000):
    print("Initializing Leviathan v2.0 Agent Simulation...")

    # 1. Setup Brain
    num_nodes = 200
    brain = LeviathanNetwork(num_nodes)

    # Define dedicated node roles based on spec
    # 5 Raycast Sensors -> Thalamus (say nodes 0 to 4)
    sensor_indices = [0, 1, 2, 3, 4]

    # 2 Motor outputs -> Cerebellum
    motor_decoder = MotorDecoder(network=brain)

    # 2. Setup Body/Environment
    arena = Arena(width=800, height=600, num_foods=15)

    # Logging
    history = {"x": [], "y": [], "atp": [], "da": [], "foods_eaten": 0}

    print(f"Starting simulation for {ticks} ticks...")
    for tick in range(ticks):

        # --- SENSORY PHASE ---
        # 1. Get raycasts from environment [5]
        proximities = arena.get_sensory_raycasts(num_rays=5, fov_deg=120)

        # 2. Map to Thalamus Input Forcing S_i(t)
        # S_i(t) = I_0 / d * sin(theta - phi_i)
        # Simplified: Inject raw proximity into the acceleration term
        for i, prox in enumerate(proximities):
            node_idx = sensor_indices[i]
            # Boost the sensory impact
            brain.phi_dot[node_idx] += prox * 50.0

        # --- COGNITIVE PHASE ---
        # 3. Step the Brain Physics (Integration, PTDP, etc)
        brain.step()
        update_metabolism(brain)

        # Calculate error for neuromodulators (simplified task error: are we moving?)
        avg_velocity = torch.mean(torch.abs(brain.phi_dot)).item()
        error = 1.0 / (avg_velocity + 0.1)

        update_neuromodulators(brain)

        # --- MOTOR PHASE ---
        # 4. Decode motors from Cerebellum
        torque_l, torque_r = motor_decoder.decode_torque()

        # 5. Move body in environment
        foods_eaten = arena.move_agent(torque_l, torque_r)

        # --- REWARD PHASE ---
        if foods_eaten > 0:
            history["foods_eaten"] += foods_eaten
            # Phase 7 embodied rewards: Food -> ATP & DA
            brain.atp += ATP_FOOD_REWARD * foods_eaten
            brain.DA += DA_REWARD_SPIKE * foods_eaten
            print(
                f"Tick {tick}: Found Food! ATP: {brain.atp.mean().item():.1f}, DA: {brain.DA:.2f}"
            )

        # --- LOGGING ---
        if tick % 50 == 0:
            history["x"].append(arena.agent_x)
            history["y"].append(arena.agent_y)
            history["atp"].append(brain.atp.mean().item())
            history["da"].append(brain.DA)

    print(f"\nSimulation Complete. Total Food Eaten: {history['foods_eaten']}")
    print(f"Final Average ATP: {history['atp'][-1]:.1f}")

    # Save trajectory for potential visualization later
    np.save("agent_trajectory.npy", {"x": history["x"], "y": history["y"]})
    print("Saved agent trajectory to agent_trajectory.npy")


if __name__ == "__main__":
    run_agent_simulation()
