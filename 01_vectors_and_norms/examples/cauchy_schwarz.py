"""
Example: Verify Cauchy-Schwarz Inequality.
Uses dot_product and l2 norm from src.
"""
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src import basic_ops, norms

def verify_cauchy_schwarz(n=1000):
    print("--- Verifying Cauchy-Schwarz: |u·v| <= ||u||·||v|| ---")
    
    successes = 0
    for _ in range(n):
        u = np.random.randn(5).tolist()
        v = np.random.randn(5).tolist()
        
        lhs = abs(basic_ops.dot_product(u, v))
        rhs = norms.l2(u) * norms.l2(v)
        
        if lhs <= rhs + 1e-12: # Tolerance for floats
            successes += 1
            
    print(f"  Trials: {n}")
    print(f"  Success rate: {successes/n * 100:.1f}%")

if __name__ == "__main__":
    verify_cauchy_schwarz()
