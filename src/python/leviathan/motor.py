import torch
from .network import LeviathanNetwork


class MotorDecoder:
    def __init__(self, network: LeviathanNetwork, kappa_motor: float = 1.0):
        self.network = network
        self.kappa = kappa_motor

        # Identify Cerebellum nodes capable of motor output
        self.cerebellum_indices = [
            i for i, label in enumerate(network.region_labels) if label == "cerebellum"
        ]

        # Split half for Left Track, half for Right Track
        half = len(self.cerebellum_indices) // 2

        self.left_motor_nodes = torch.tensor(
            self.cerebellum_indices[:half], dtype=torch.long
        )
        self.right_motor_nodes = torch.tensor(
            self.cerebellum_indices[half:], dtype=torch.long
        )

    def decode_torque(self):
        """
        Decodes physical torque from the designated motor nodes.
        Only nodes currently in the "forward swinging" phase (cos(phi) > 0) contribute.
        """
        if len(self.left_motor_nodes) == 0 or len(self.right_motor_nodes) == 0:
            return 0.0, 0.0

        phi_left = self.network.phi[self.left_motor_nodes]
        phi_dot_left = self.network.phi_dot[self.left_motor_nodes]

        phi_right = self.network.phi[self.right_motor_nodes]
        phi_dot_right = self.network.phi_dot[self.right_motor_nodes]

        # Mask: cos(phi) > 0
        mask_left = (torch.cos(phi_left) > 0).float()
        mask_right = (torch.cos(phi_right) > 0).float()

        # ---- Phase 15 Behavioral Inhibition ----
        # High SER (Serotonin) -> Behavioral Persistence / Inhibition (Lower gain)
        # High ACH (Acetylcholine) -> High Arousal / Exploration (Higher gain)
        ser = float(self.network.SER)
        ach = float(self.network.ACH)

        # Inhibition factor: scales gain down as SER increases, up as ACH increases
        inhibition = ach / (1.0 + ser)
        effective_kappa = self.kappa * inhibition

        torque_left = effective_kappa * torch.sum(mask_left * phi_dot_left).item()
        torque_right = effective_kappa * torch.sum(mask_right * phi_dot_right).item()

        return torque_left, torque_right
