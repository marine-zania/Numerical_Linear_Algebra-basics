"""
Demo: Basic Vector Operations
Shows add, subtract, scalar multiply, and dot/cross products.
"""
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src import basic_ops

def run_demo():
    u = [1.0, 2.0, 3.0]
    v = [4.0, 5.0, 6.0]
    alpha = 0.5
    
    print("--- Vectors ---")
    print(f"  u = {u}")
    print(f"  v = {v}")
    print(f"  α = {alpha}\n")
    
    print("--- Operations ---")
    print(f"  u + v      = {basic_ops.add(u, v)}")
    print(f"  u - v      = {basic_ops.subtract(u, v)}")
    print(f"  α * u      = {basic_ops.scalar_multiply(alpha, u)}")
    print(f"  u · v      = {basic_ops.dot_product(u, v)}")
    print(f"  u × v (3D) = {basic_ops.cross_product_3d(u, v)}")

if __name__ == "__main__":
    run_demo()
