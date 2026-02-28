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
    def __init__(self, width=800, height=600, num_foods=20, walls=None):
        self.width = width
        self.height = height

        # Agent state
        self.agent_x = width / 2
        self.agent_y = height / 2
        self.agent_heading = 0.0
        self.agent_radius = 15.0

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
    def move_agent(self, torque_left, torque_right):
        """
        Differential drive kinematics. Torques treated as wheel velocities (px/step).
        Returns count of food items eaten this step.
        """
        L = 30.0
        v_l = float(torque_left)
        v_r = float(torque_right)

        v = (v_r + v_l) / 2.0
        w = (v_r - v_l) / L

        self.agent_heading = (self.agent_heading + w) % (2 * math.pi)
        nx = self.agent_x + v * math.cos(self.agent_heading)
        ny = self.agent_y + v * math.sin(self.agent_heading)

        # Wall collision: slide along axis
        if not self._point_inside_wall(nx, self.agent_y):
            self.agent_x = nx
        if not self._point_inside_wall(self.agent_x, ny):
            self.agent_y = ny

        # Boundary wrap
        self.agent_x = self.agent_x % self.width
        self.agent_y = self.agent_y % self.height

        return self.check_collisions()

    def check_collisions(self):
        eaten_count = 0
        remaining = []
        thresh2 = (self.agent_radius + self.food_radius) ** 2
        for fx, fy in self.foods:
            dx = self.agent_x - fx
            dy = self.agent_y - fy
            if dx * dx + dy * dy <= thresh2:
                eaten_count += 1
                self.spawn_food()
            else:
                remaining.append((fx, fy))
        self.foods = remaining
        return eaten_count

    # ── Raycasting ───────────────────────────────────────────────────────
    def get_sensory_raycasts(self, num_rays=5, fov_deg=120, max_dist=250.0):
        """
        Returns (food_proximities, wall_proximities), each shape (num_rays,).
        Food:  1 = food touching agent, 0 = nothing in range
        Wall:  1 = wall at agent face,  0 = clear
        """
        food_prox = np.zeros(num_rays, dtype=np.float32)
        wall_prox = np.zeros(num_rays, dtype=np.float32)

        fov_rad = math.radians(fov_deg)
        angles = np.linspace(-fov_rad / 2, fov_rad / 2, num_rays)

        for i, offset in enumerate(angles):
            ray_angle = self.agent_heading + offset
            rdx = math.cos(ray_angle)
            rdy = math.sin(ray_angle)

            # Food hit
            closest_food = max_dist
            for fx, fy in self.foods:
                vx, vy = fx - self.agent_x, fy - self.agent_y
                proj = vx * rdx + vy * rdy
                if 0 < proj < closest_food:
                    ortho = abs(vx * (-rdy) + vy * rdx)
                    if ortho <= self.food_radius:
                        closest_food = proj
            if closest_food < max_dist:
                food_prox[i] = 1.0 - closest_food / max_dist

            # Wall hit (step-march)
            closest_wall = max_dist
            step = 5.0
            dist = 0.0
            while dist < max_dist:
                px = self.agent_x + rdx * dist
                py = self.agent_y + rdy * dist
                if self._point_inside_wall(px, py):
                    closest_wall = dist
                    break
                dist += step
            if closest_wall < max_dist:
                wall_prox[i] = 1.0 - closest_wall / max_dist

        return food_prox, wall_prox

    # ── Serialisation (for dashboard) ───────────────────────────────────
    def to_dict(self, brain):
        return {
            "agent": {
                "x": round(self.agent_x, 1),
                "y": round(self.agent_y, 1),
                "h": round(self.agent_heading, 3),
                "r": self.agent_radius,
            },
            "foods": [{"x": round(x, 1), "y": round(y, 1)} for x, y in self.foods],
            "walls": [w.to_dict() for w in self.walls],
            "stats": {
                "atp": round(float(brain.atp.mean()), 1),
                "da": round(float(brain.DA), 3),
                "na": round(float(brain.NA), 3),
                "ser": round(float(brain.SER), 3),
                "ach": round(float(brain.ACH), 3),
            },
        }
