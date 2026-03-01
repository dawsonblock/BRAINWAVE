import torch
import random
import copy


class Genome:
    """
    Encodes the 'innate' parameters of a Leviathan network.
    These are traits that are passed down and modified via evolution,
    not learned via plasticity during an agent's lifetime.
    """

    def __init__(self, data=None):
        if data is not None:
            self.data = data
        else:
            # Default 'Root' Genome
            self.data = {
                "eta_plus": 0.05,
                "eta_minus": 0.04,
                "gwt_k": 10,
                "gwt_gain": 0.5,
                "initial_weight_std": 0.1,
                "atp_decay_rate": 0.1,
                "motor_gain": 1.0,
            }

    def mutate(self, rate=0.1, sigma=0.05):
        """Randomly jitter parameters."""
        new_data = copy.deepcopy(self.data)
        for key in new_data:
            if random.random() < rate:
                if isinstance(new_data[key], float):
                    new_data[key] += random.gauss(0, sigma)
                    new_data[key] = max(1e-4, new_data[key])
                elif isinstance(new_data[key], int):
                    new_data[key] += random.choice([-1, 0, 1])
                    new_data[key] = max(1, new_data[key])
        return Genome(new_data)

    @staticmethod
    def crossover(parent_a, parent_b):
        """Uniform crossover between two genomes."""
        child_data = {}
        for key in parent_a.data:
            child_data[key] = random.choice([parent_a.data[key], parent_b.data[key]])
        return Genome(child_data)


class EvoController:
    """
    Manages the population life-cycle and selection.
    """

    def __init__(self, population_size):
        self.population_size = population_size
        self.generation = 0
        self.best_fitness = 0.0

    def select_and_reproduce(self, population_fitness, genomes):
        """
        Tournament selection: Pick the best performers and generate next gen.
        """
        self.generation += 1
        new_genomes = []

        # Sort by fitness
        sorted_indices = sorted(
            range(len(population_fitness)),
            key=lambda i: population_fitness[i],
            reverse=True,
        )

        self.best_fitness = population_fitness[sorted_indices[0]]

        # Elites (top 20% carry over directly)
        num_elites = max(1, self.population_size // 5)
        for i in range(num_elites):
            new_genomes.append(genomes[sorted_indices[i]])

        # Fill the rest with children of elites
        while len(new_genomes) < self.population_size:
            p1_idx = random.choice(sorted_indices[:num_elites])
            p2_idx = random.choice(sorted_indices[:num_elites])

            child = Genome.crossover(genomes[p1_idx], genomes[p2_idx])
            child = child.mutate()
            new_genomes.append(child)

        return new_genomes
