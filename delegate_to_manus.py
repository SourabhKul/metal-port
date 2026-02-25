#!/usr/bin/env python3
import json
import requests
import sys
import os

# Manus API Configuration
API_KEY = "sk--WPHhbtI2vchXvG_77OEVq9q7ll4GwLZMYSEbHKd4qxiCkD3ZdDXeilh-43iswclETYAV9DYR2xgmDHUPAlULs0Yz_3C"
API_URL = "https://api.manus.ai/v1/tasks"

def create_manus_task(prompt):
    headers = {
        "API_KEY": API_KEY,
        "Content-Type": "application/json"
    }
    
    data = {
        "prompt": prompt,
        "agentProfile": "manus-1.6-max"  # Using max for research tasks
    }
    
    print(f"Triggering Manus Task: {prompt[:100]}...")
    
    try:
        response = requests.post(API_URL, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        print(f"Success! Task ID: {result.get('task_id')}")
        print(f"Task URL: {result.get('task_url')}")
        return result
    except Exception as e:
        print(f"Error triggering Manus task: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"Response: {e.response.text}")
        return None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        task_prompt = sys.argv[1]
    else:
        context = """
        We have developed a high-performance C++ Metal SBI (Simulation-Based Inference) solver running on an M4 Max Mac Studio.
        
        Current Achievements:
        - Performance: ~15.8 Billion particle-updates per second (12x speedup over Python/MLX).
        - Application: Tokamak Plasma Turbulence simulation (Lorentz Force integration).
        - Algorithm: ABC-SMC (Approximate Bayesian Computation Sequential Monte Carlo) with Beta-distributed step sizes for parameter refinement.
        - Successfully converged on optimal magnetic shim configurations with extreme precision (3e-4 distance).
        
        Goal:
        Find out how we can do publishing-worthy research using this amazing solver. 
        Identify specific grand challenges in fusion physics, plasma turbulence, or SBI methodology where this 12x speedup on consumer hardware allows us to perform research that was previously limited to supercomputers.
        Suggest specific papers or journals (e.g., Physical Review Letters, Journal of Computational Physics, Nuclear Fusion) that would value these results.
        """
        task_prompt = f"Research Plan for C++ Metal SBI Solver:\n{context}"
    
    create_manus_task(task_prompt)
