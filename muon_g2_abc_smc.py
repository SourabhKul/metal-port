import numpy as np
import subprocess
import time

class G2_ABC_SMC_Controller:
    def __init__(self, num_populations=100, generations=10):
        self.num_populations = num_populations
        self.generations = generations
        # Parameters to infer: 
        # 1. B_y field bias (ppm offset)
        # 2. Kicker amplitude (arbitrary scale)
        # 3. Kicker decay constant (micro-seconds)
        self.population = np.random.uniform(-10.0, 10.0, (num_populations, 3)) 

    def run_generation(self, gen_idx):
        print(f"--- Muon g-2 Inference Gen {gen_idx} ---")
        
        # In a real run, we pass these to the muon_g2_tracker binary
        # For now, we simulate the 'Observation Comparison'
        # Target (Ground Truth): [0.5 ppm bias, 1.2 kicker amp, 140.0ms decay]
        target = np.array([0.5, 1.2, 140.0])
        
        distances = np.linalg.norm(self.population - target, axis=1)
        fitness = 1.0 / (distances + 1e-6)
        
        best_idx = np.argmax(fitness)
        print(f"  Best Systematic Bias Estimate: {self.population[best_idx]}")
        
        # Perturbation using Beta Distribution (as requested by PI)
        probabilities = fitness / np.sum(fitness)
        indices = np.random.choice(len(self.population), size=self.num_populations, p=probabilities)
        
        new_pop = []
        scale = 1.0 * (0.8 ** gen_idx)
        for idx in indices:
            parent = self.population[idx]
            # Apply Beta-distributed perturbation
            child = parent + (np.random.beta(5, 5, size=3) - 0.5) * scale
            new_pop.append(child)
            
        self.population = np.array(new_pop)

if __name__ == "__main__":
    ctrl = G2_ABC_SMC_Controller()
    for i in range(10):
        ctrl.run_generation(i)
