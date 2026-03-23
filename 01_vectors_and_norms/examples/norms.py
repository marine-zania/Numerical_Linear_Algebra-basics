"""
Demo: Vector Norms
Shows L1, L2, Linf, and general Lp norm computations.
"""
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src import norms

def run_demo():
    v = [3.0, -4.0, 0.0]
    
    print(f"--- Vector: {v} ---")
    print(f"  L1   norm: {norms.l1(v)}")
    print(f"  L2   norm: {norms.l2(v):.4f}")
    print(f"  Linf norm: {norms.linf(v)}")
    print(f"  L3   norm: {norms.lp(v, 3):.4f}\n")
    
    u_normed = norms.normalize(v, 'l2')
    print(f"  Normalized (L2): {u_normed}")
    print(f"  New L2 norm:     {norms.l2(u_normed):.4f}")

if __name__ == "__main__":
    run_demo()
