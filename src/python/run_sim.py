import torch
from leviathan.network import LeviathanNetwork
from leviathan.plasticity import apply_ptdp
from leviathan.motor import MotorDecoder
from leviathan.config import DT


def main():
    print("Initializing Leviathan PyTorch Engine: 100 Nodes...")
    net = LeviathanNetwork(num_nodes=100)

    # Initialize Motor Decoder
    motor = MotorDecoder(net, kappa_motor=2.5)

    from leviathan.endocrine import update_metabolism, update_neuromodulators

    # Simulation loop
    ticks = 500
    print(f"Running {ticks} ticks of DDE integration (\u0394t={DT}s)...")

    for _ in range(ticks):
        # 1. Integrate Phase/Velocity using Ring Buffers
        delayed_phi = net.step()

        # 2. Apply Plasticity based on phase deltas
        net.weights = apply_ptdp(net.weights, net.phi, delayed_phi, net.DA)

        # 3. Endocrine & Metabolism
        update_metabolism(net)
        update_neuromodulators(net)

    # Decode Final Torque
    t_left, t_right = motor.decode_torque()

    print("\nSimulation Complete.")
    print(f"Final Phase Velocity (mean): {net.phi_dot.mean():.4f} rad/s")
    print(f"Final Mean ATP: {net.atp.mean():.2f} (Target: 800.0)")
    print(f"Total Active Synapses: {(net.weights > 0).sum().item()}")
    print("\nEndocrine State:")
    print(
        f"  DA: {net.DA:.3f} | NA: {net.NA:.3f} | SER: {net.SER:.3f} | ACH: {net.ACH:.3f}"
    )
    print("\nMotor Output:")
    print(f"  Left Track Torque:  {t_left:.3f}")
    print(f"  Right Track Torque: {t_right:.3f}")


if __name__ == "__main__":
    main()
