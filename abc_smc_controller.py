import numpy as np
import subprocess
import time

class ABCSMCController:
    def __init__(self, num_populations=10, particles_per_pop=1000, generations=5):
        self.num_populations = num_populations
        self.particles_per_pop = particles_per_pop
        self.generations = generations
        # Prior: Uniform [-0.05, 0.05] for B_x, B_y, B_z offsets
        self.population = np.random.uniform(-0.05, 0.05, (num_populations, 3))
        self.weights = np.ones(num_populations) / num_populations

    def sample_beta_perturbation(self, current_val, scale=0.01):
        # Beta distribution perturbation [0, 1] shifted to [-scale, scale]
        # Using alpha=2, beta=2 for a bell-ish shape on [0, 1]
        shift = np.random.beta(2, 2)
        return current_val + (shift - 0.5) * 2 * scale

    def run_generation(self, gen_idx):
        print(f"--- Generation {gen_idx} ---")
        # In a real implementation, we would write these params to a buffer 
        # and call the metal_sbi binary. For now, we simulate the loop logic.
        
        # Simulated 'fitness' based on a hidden optimal shim [0.01, -0.02, 0.005]
        optimal = np.array([0.01, -0.02, 0.005])
        distances = np.linalg.norm(self.population - optimal, axis=1)
        fitness = 1.0 / (distances + 1e-6)
        
        # Selection & Perturbation for next gen
        probabilities = fitness / np.sum(fitness)
        indices = np.random.choice(len(self.population), size=self.num_populations, p=probabilities)
        
        new_pop = []
        for idx in indices:
            parent = self.population[idx]
            child = [self.sample_beta_perturbation(p) for p in parent]
            new_pop.append(child)
            
        self.population = np.array(new_pop)
        print(f"Top Candidate: {self.population[np.argmax(fitness)]}")

if __name__ == "__main__":
    controller = ABCSMCController()
    for g in range(5):
        controller.run_generation(g)
