import math
import random
import numpy as np


class Wall:
    """Axis-aligned rectangular wall."""

    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.x2, self.y2 = x + w, y + h

    def to_dict(self):
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


class Arena:
    def __init__(
        self, width=800, height=600, num_foods=20, num_agents=1, walls=None, seed=None
    ):
        if seed is not None:
            random.seed(seed)
        self.width = width
        self.height = height
        self.num_agents = num_agents

        # Multi-Agent state
        self.agents_x = [width / 2 + random.uniform(-50, 50) for _ in range(num_agents)]
        self.agents_y = [
            height / 2 + random.uniform(-50, 50) for _ in range(num_agents)
        ]
        self.agents_heading = [
            random.uniform(0, 2 * math.pi) for _ in range(num_agents)
        ]
        self.agent_radius = 15.0

        # Pheromone Grid (Reward & Danger trails)
        # Resolution: 40x30 for 800x600 arena (20px per cell)
        self.grid_res = 20
        self.grid_w = width // self.grid_res
        self.grid_h = height // self.grid_res
        self.pheromones_reward = np.zeros((self.grid_w, self.grid_h), dtype=np.float32)
        self.pheromones_danger = np.zeros((self.grid_w, self.grid_h), dtype=np.float32)

        # Walls (8D: obstacle avoidance)
        self.walls: list[Wall] = walls if walls else []

        # Food targets
        self.foods = []
        self.food_radius = 12.0
        for _ in range(num_foods):
            self.spawn_food()

    # ── Walls helper ─────────────────────────────────────────────────────
    def add_default_obstacles(self):
        """Add a handful of interior rectangular obstacles."""
        self.walls = [
            Wall(150, 100, 120, 20),
            Wall(500, 150, 20, 130),
            Wall(300, 350, 200, 20),
            Wall(100, 400, 20, 150),
            Wall(600, 350, 150, 20),
        ]

    def _point_inside_wall(self, x, y):
        for w in self.walls:
            if w.x <= x <= w.x2 and w.y <= y <= w.y2:
                return True
        return False

    # ── Food ─────────────────────────────────────────────────────────────
    def spawn_food(self):
        for _ in range(50):  # retry until clear of walls
            x = random.uniform(self.food_radius, self.width - self.food_radius)
            y = random.uniform(self.food_radius, self.height - self.food_radius)
            if not self._point_inside_wall(x, y):
                self.foods.append((x, y))
                return
        # fallback — accept anywhere
        self.foods.append(
            (
                random.uniform(self.food_radius, self.width - self.food_radius),
                random.uniform(self.food_radius, self.height - self.food_radius),
            )
        )

    # ── Kinematics ───────────────────────────────────────────────────────
    def move_agent(self, agent_idx, torque_left, torque_right):
        """
        Differential drive kinematics for a specific agent.
        Returns count of food items eaten by this agent.
        """
        L = 30.0
        v_l = float(torque_left)
        v_r = float(torque_right)

        v = (v_r + v_l) / 2.0
        w = (v_r - v_l) / L

        # Decay pheromones globally (called once per simulation step ideally,
        # but here we do it per agent move for simplicity, scaled down)
        self.pheromones_reward *= 0.999
        self.pheromones_danger *= 0.999

        h = self.agents_heading[agent_idx]
        self.agents_heading[agent_idx] = (h + w) % (2 * math.pi)

        nx = self.agents_x[agent_idx] + v * math.cos(self.agents_heading[agent_idx])
        ny = self.agents_y[agent_idx] + v * math.sin(self.agents_heading[agent_idx])

        # Wall collision: slide along axis
        if not self._point_inside_wall(nx, self.agents_y[agent_idx]):
            self.agents_x[agent_idx] = nx
        if not self._point_inside_wall(self.agents_x[agent_idx], ny):
            self.agents_y[agent_idx] = ny

        # Agent-Agent collision
        for i in range(self.num_agents):
            if i == agent_idx:
                continue
            dx = self.agents_x[agent_idx] - self.agents_x[i]
            dy = self.agents_y[agent_idx] - self.agents_y[i]
            dist_sq = dx * dx + dy * dy
            min_dist = self.agent_radius * 2
            if dist_sq < min_dist * min_dist:
                # Push back slightly
                dist = math.sqrt(dist_sq) + 1e-6
                overlap = min_dist - dist
                self.agents_x[agent_idx] += (dx / dist) * overlap * 0.5
                self.agents_y[agent_idx] += (dy / dist) * overlap * 0.5

        # Boundary wrap
        self.agents_x[agent_idx] = self.agents_x[agent_idx] % self.width
        self.agents_y[agent_idx] = self.agents_y[agent_idx] % self.height

        return self.check_collisions(agent_idx)

    def check_collisions(self, agent_idx):
        eaten_count = 0
        remaining = []
        ax, ay = self.agents_x[agent_idx], self.agents_y[agent_idx]
        thresh2 = (self.agent_radius + self.food_radius) ** 2
        for fx, fy in self.foods:
            dx = ax - fx
            dy = ay - fy
            if dx * dx + dy * dy <= thresh2:
                eaten_count += 1
                self.spawn_food()
                # Drop reward pheromone at collision site
                gx = int(ax / self.grid_res) % self.grid_w
                gy = int(ay / self.grid_res) % self.grid_h
                self.pheromones_reward[gx, gy] = min(
                    self.pheromones_reward[gx, gy] + 1.0, 5.0
                )
            else:
                remaining.append((fx, fy))
        self.foods = remaining
        return eaten_count

    # ── Raycasting ───────────────────────────────────────────────────────
    def get_sensory_raycasts(self, agent_idx, num_rays=5, fov_deg=120, max_dist=250.0):
        """
        Returns (food_proximities, wall_proximities, peer_proximities), each shape (num_rays,).
        """
        food_prox = np.zeros(num_rays, dtype=np.float32)
        wall_prox = np.zeros(num_rays, dtype=np.float32)
        peer_prox = np.zeros(num_rays, dtype=np.float32)

        ax, ay = self.agents_x[agent_idx], self.agents_y[agent_idx]
        ah = self.agents_heading[agent_idx]

        fov_rad = math.radians(fov_deg)
        angles = np.linspace(-fov_rad / 2, fov_rad / 2, num_rays)

        for i, offset in enumerate(angles):
            ray_angle = ah + offset
            rdx = math.cos(ray_angle)
            rdy = math.sin(ray_angle)

            # Food hit
            closest_food = max_dist
            for fx, fy in self.foods:
                vx, vy = fx - ax, fy - ay
                proj = vx * rdx + vy * rdy
                if 0 < proj < closest_food:
                    ortho = abs(vx * (-rdy) + vy * rdx)
                    if ortho <= self.food_radius:
                        closest_food = proj
            if closest_food < max_dist:
                food_prox[i] = 1.0 - closest_food / max_dist

            # Peer hit
            closest_peer = max_dist
            for p_idx in range(self.num_agents):
                if p_idx == agent_idx:
                    continue
                vx, vy = self.agents_x[p_idx] - ax, self.agents_y[p_idx] - ay
                proj = vx * rdx + vy * rdy
                if 0 < proj < closest_peer:
                    ortho = abs(vx * (-rdy) + vy * rdx)
                    if ortho <= self.agent_radius:
                        closest_peer = proj
            if closest_peer < max_dist:
                peer_prox[i] = 1.0 - closest_peer / max_dist

            # Wall hit (step-march)
            closest_wall = max_dist
            step = 5.0
            dist = 0.0
            while dist < max_dist:
                px = ax + rdx * dist
                py = ay + rdy * dist
                if self._point_inside_wall(px, py):
                    closest_wall = dist
                    break
                dist += step
            if closest_wall < max_dist:
                wall_prox[i] = 1.0 - closest_wall / max_dist

        return food_prox, wall_prox, peer_prox

    # ── Chemical Gradient (Scents) ───────────────────────────────────────
    def get_chemical_gradient(self, agent_idx):
        """
        Returns (food_scent, reward_pheromone, danger_pheromone) intensities.
        """
        ax, ay = self.agents_x[agent_idx], self.agents_y[agent_idx]

        # 1. Food Scent (Inverse Square)
        food_intensity = 0.0
        epsilon = 100.0
        for fx, fy in self.foods:
            dx, dy = ax - fx, ay - fy
            food_intensity += 1.0 / (dx * dx + dy * dy + epsilon)
        food_scent = min(food_intensity * 10.0, 1.0)

        # 2. Pheromones (Grid Sampling)
        gx, gy = (
            int(ax / self.grid_res) % self.grid_w,
            int(ay / self.grid_res) % self.grid_h,
        )
        reward_scent = float(self.pheromones_reward[gx, gy])
        danger_scent = float(self.pheromones_danger[gx, gy])

        return food_scent, min(reward_scent, 1.0), min(danger_scent, 1.0)

    # ── Serialisation (for dashboard) ───────────────────────────────────
    def to_dict(self, brains):
        """brains is a list of LeviathanNetwork objects."""
        agent_list = []
        for i in range(self.num_agents):
            b = brains[i]
            agent_list.append(
                {
                    "id": i,
                    "x": round(self.agents_x[i], 1),
                    "y": round(self.agents_y[i], 1),
                    "h": round(self.agents_heading[i], 3),
                    "r": self.agent_radius,
                    "stats": {
                        "atp": round(float(b.atp.mean()), 1),
                        "da": round(float(b.DA), 3),
                        "na": round(float(b.NA), 3),
                        "ser": round(float(b.SER), 3),
                        "ach": round(float(b.ACH), 3),
                    },
                }
            )
        return {
            "agents": agent_list,
            "foods": [{"x": round(x, 1), "y": round(y, 1)} for x, y in self.foods],
            "walls": [w.to_dict() for w in self.walls],
        }
