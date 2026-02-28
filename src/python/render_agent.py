import numpy as np
import matplotlib.pyplot as plt

data = np.load("agent_trajectory.npy", allow_pickle=True).item()
x = data["x"]
y = data["y"]

plt.figure(figsize=(8, 6))
plt.plot(x, y, label="Agent Path", color="b", alpha=0.7)
plt.scatter(x[0], y[0], color="g", label="Start", zorder=5)
plt.scatter(x[-1], y[-1], color="r", label="End", zorder=5)

plt.xlim(0, 800)
plt.ylim(0, 600)
plt.title("Leviathan v2.0 Embodied Agent Trajectory")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.legend()
plt.grid(True)
plt.savefig("trajectory.png")
print("Successfully generated trajectory.png")
