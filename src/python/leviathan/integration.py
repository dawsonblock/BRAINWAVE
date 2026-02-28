"""integration.py — Shared RK2 integrator (Phase 10)

Extracted from network.py so Python and CUDA converge on identical math.
Both backends call integrate_rk2() — ensures numerical parity.
"""

import torch


def integrate_rk2(phi, phi_dot, accel_fn, dt: float, v_max: float):
    """
    Midpoint (RK2) integration step.

    Args:
        phi      : (N,) current phase
        phi_dot  : (N,) current velocity
        accel_fn : callable(phi, phi_dot) → acceleration (N,)
        dt       : timestep
        v_max    : velocity clamp

    Returns:
        phi_new, phi_dot_new  both (N,)
    """
    a1 = accel_fn(phi, phi_dot)
    phi_dot_mid = phi_dot + 0.5 * dt * a1
    phi_mid = phi + 0.5 * dt * phi_dot

    a2 = accel_fn(phi_mid, phi_dot_mid)
    phi_dot_new = (phi_dot + dt * a2).clamp(-v_max, v_max)
    phi_new = (phi + dt * phi_dot_mid) % (2 * torch.pi)

    return phi_new, phi_dot_new
