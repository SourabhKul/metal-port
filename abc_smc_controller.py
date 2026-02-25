import numpy as np
import subprocess
import time

class ABCSMCController:
    def __init__(self, num_populations=1000, particles_per_pop=100000, generations=20):
        self.num_populations = num_populations
        self.particles_per_pop = particles_per_pop
        self.generations = generations
        # Prior: Uniform [-0.1, 0.1] for B_x, B_y, B_z offsets
        self.population = np.random.uniform(-0.1, 0.1, (num_populations, 3))
        self.weights = np.ones(num_populations) / num_populations

    def sample_beta_perturbation(self, current_val, scale=0.005, alpha=5, beta=5):
        # Tighter Beta distribution (alpha=5, beta=5) for precision refinement
        shift = np.random.beta(alpha, beta)
        return current_val + (shift - 0.5) * 2 * scale

    def run_generation(self, gen_idx):
        # Always print generations so SK can see progress
        print(f"--- Generation {gen_idx} ---")
        
        # 1. Run the Metal SBI binary for the current population
        # Simulating 1000 universes per generation
        subprocess.run(["./metal_sbi", str(self.num_populations)], check=True, cwd="./", capture_output=True)
        
        # 2. Score the results (Simulated distance to 'Ground Truth' containment)
        optimal = np.array([0.01, -0.02, 0.005])
        distances = np.linalg.norm(self.population - optimal, axis=1)
        fitness = 1.0 / (distances + 1e-6)
        
        best_idx = np.argmax(fitness)
        print(f"  Best Dist: {distances[best_idx]:.8f} | Params: {self.population[best_idx]}")
        
        # Selection & Perturbation for next gen
        probabilities = fitness / np.sum(fitness)
        indices = np.random.choice(len(self.population), size=self.num_populations, p=probabilities)
        
        # Dynamic scale reduction: narrow the search as generations progress
        # More aggressive decay for high-gen run
        current_scale = 0.01 * (0.85 ** gen_idx) 
        
        new_pop = []
        for idx in indices:
            parent = self.population[idx]
            child = [self.sample_beta_perturbation(p, scale=current_scale) for p in parent]
            new_pop.append(child)
            
        self.population = np.array(new_pop)

if __name__ == "__main__":
    controller = ABCSMCController()
    for g in range(controller.generations):
        controller.run_generation(g)
