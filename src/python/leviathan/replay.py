"""
replay.py — Phase 10: Deterministic replay logger

Records mean phi/phi_dot per tick to a JSON log.
Useful for verifying determinism across runs and debugging divergence.
"""

import json
import torch


class DeterministicReplay:
    def __init__(self, filename: str):
        self.filename = filename
        self.log = []

    def record(self, step: int, phi: torch.Tensor, phi_dot: torch.Tensor):
        self.log.append(
            {
                "step": step,
                "phi_mean": round(float(phi.mean()), 8),
                "phi_dot_mean": round(float(phi_dot.mean()), 8),
                "phi_std": round(float(phi.std()), 8),
            }
        )

    def save(self):
        with open(self.filename, "w") as f:
            json.dump(self.log, f, indent=2)
        print(f"[replay] Saved {len(self.log)} steps → {self.filename}")

    def compare(self, other_file: str, atol: float = 1e-6) -> bool:
        """Compare two replay logs for determinism verification."""
        with open(other_file) as f:
            other = json.load(f)
        mismatches = 0
        for a, b in zip(self.log, other):
            if abs(a["phi_mean"] - b["phi_mean"]) > atol:
                print(
                    f"[replay] Mismatch at step {a['step']}: {a['phi_mean']} vs {b['phi_mean']}"
                )
                mismatches += 1
        if mismatches == 0:
            print(f"[replay] ✅ Logs match (atol={atol})")
        return mismatches == 0
