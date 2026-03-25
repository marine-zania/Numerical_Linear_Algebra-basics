"""
Basic Matrix Operations
Shows add, subtract, scalar multiply, transpose, and trace.
"""
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src import matrix_ops

def run_example():
    A = [[1.0, 2.0], [3.0, 4.0]]
    B = [[5.0, 6.0], [7.0, 8.0]]
    alpha = 2.0
    
    print("--- Matrices ---")
    print(f"  A = {A}")
    print(f"  B = {B}")
    print(f"  α = {alpha}\n")
    
    print("--- Operations ---")
    print(f"  A + B      = {matrix_ops.add(A, B)}")
    print(f"  A - B      = {matrix_ops.subtract(A, B)}")
    print(f"  α * A      = {matrix_ops.scalar_multiply(alpha, A)}")
    print(f"  A^T        = {matrix_ops.transpose(A)}")
    print(f"  tr(A)      = {matrix_ops.trace(A)}")

if __name__ == "__main__":
    run_example()
