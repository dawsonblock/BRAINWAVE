import math
import random
import numpy as np


class Arena:
    def __init__(self, width=800, height=600, num_foods=10):
        self.width = width
        self.height = height

        # Agent state
        self.agent_x = width / 2
        self.agent_y = height / 2
        self.agent_heading = 0.0  # Radians
        self.agent_radius = 15.0

        # Food targets
        self.foods = []
        self.food_radius = 12.0  # larger for easier collision
        for _ in range(num_foods):
            self.spawn_food()

    def spawn_food(self):
        """Spawns a new food item at a random location."""
        x = random.uniform(self.food_radius, self.width - self.food_radius)
        y = random.uniform(self.food_radius, self.height - self.food_radius)
        self.foods.append((x, y))

    def move_agent(self, torque_left, torque_right):
        """
        Updates agent kinematics based on tank-drive torques.
        Torque values are treated as wheel velocities in px/step.
        Returns count of food items eaten this step.
        """
        L = 30.0  # axle width in pixels
        v_l = float(torque_left)
        v_r = float(torque_right)

        v = (v_r + v_l) / 2.0
        w = (v_r - v_l) / L

        self.agent_heading += w
        self.agent_heading = self.agent_heading % (2 * math.pi)

        self.agent_x += v * math.cos(self.agent_heading)
        self.agent_y += v * math.sin(self.agent_heading)

        # Boundary wrap (keeps agent in arena)
        self.agent_x = self.agent_x % self.width
        self.agent_y = self.agent_y % self.height

        return self.check_collisions()

    def check_collisions(self):
        """Checks if agent overlaps with any food. Removes food and returns count."""
        eaten_count = 0
        remaining_foods = []

        collision_dist2 = (self.agent_radius + self.food_radius) ** 2

        for fx, fy in self.foods:
            dx = self.agent_x - fx
            dy = self.agent_y - fy
            dist2 = dx * dx + dy * dy
            if dist2 <= collision_dist2:
                eaten_count += 1
                self.spawn_food()  # Respawn immediately to keep environment populated
            else:
                remaining_foods.append((fx, fy))

        self.foods = remaining_foods
        return eaten_count

    def get_sensory_raycasts(self, num_rays=5, fov_deg=90, max_dist=200.0):
        """
        Casts rays from the agent to detect food.
        Returns an array of normalized proximities [0, 1].
        1 = Food touching agent, 0 = No food in range.
        """
        proximities = np.zeros(num_rays, dtype=np.float32)
        fov_rad = math.radians(fov_deg)

        # Angles relative to heading
        angles = np.linspace(-fov_rad / 2, fov_rad / 2, num_rays)

        for i, angle_offset in enumerate(angles):
            ray_angle = self.agent_heading + angle_offset

            # Find closest food intersecting this ray (simplified circle-line intersection or point-distance)
            closest_dist = max_dist
            ray_dx = math.cos(ray_angle)
            ray_dy = math.sin(ray_angle)

            for fx, fy in self.foods:
                # Vector to food
                vx = fx - self.agent_x
                vy = fy - self.agent_y

                # Project onto ray
                projection = vx * ray_dx + vy * ray_dy

                if projection > 0 and projection < closest_dist:
                    # Orthogonal distance to ray
                    ortho_dist = abs(vx * (-ray_dy) + vy * ray_dx)

                    if ortho_dist <= self.food_radius:  # Hit
                        closest_dist = projection

            if closest_dist < max_dist:
                # Normalize 0 to 1 (1 being closest)
                proximities[i] = 1.0 - (closest_dist / max_dist)

        return proximities
