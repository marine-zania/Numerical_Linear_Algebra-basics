"""
Demo: Gram-Schmidt Orthonormalization
Orthonormalize a set of linearly independent vectors.
"""
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src import orthogonalization, basic_ops, norms

def run_demo():
    # A set of 3 linearly independent vectors in R3
    vectors = [
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1]
    ]
    
    print("--- Original Vectors ---")
    for i, v in enumerate(vectors):
        print(f"  v{i}: {v}")
    
    # Run GS!
    ortho = orthogonalization.gram_schmidt(vectors)
    
    print("\n--- Orthonormalized Vectors (q) ---")
    for i, q in enumerate(ortho):
        q_str = [f"{x:+.4f}" for x in q]
        print(f"  q{i}: {q_str} (norm: {norms.l2(q):.2f})")
    
    # Check orthogonality
    print("\n--- Orthogonality Check (q0 · q1) ---")
    dot = basic_ops.dot_product(ortho[0], ortho[1])
    print(f"  q0 · q1 = {dot:+.12f} (should be approx 0)")

if __name__ == "__main__":
    run_demo()
