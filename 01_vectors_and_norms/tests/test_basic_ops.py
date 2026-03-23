"""Verifying basic vector operations against NumPy."""
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src import basic_ops

def test_add():
    u, v = [1.0, -2.0, 3.0], [4.0, 0.0, -1.0]
    ours = basic_ops.add(u, v)
    theirs = (np.array(u) + np.array(v)).tolist()
    
    print(f"  Add: {u} + {v}")
    print(f"    - Ours:  {ours}")
    print(f"    - NumPy: {theirs}")
    assert np.allclose(ours, theirs)
    print("    ✓ VERIFIED MATCH\n")

def test_subtract():
    u, v = [1.0, -2.0, 3.0], [4.0, 0.0, -1.0]
    ours = basic_ops.subtract(u, v)
    theirs = (np.array(u) - np.array(v)).tolist()
    
    print(f"  Subtract: {u} - {v}")
    print(f"    - Ours:  {ours}")
    print(f"    - NumPy: {theirs}")
    assert np.allclose(ours, theirs)
    print("    ✓ VERIFIED MATCH\n")

def test_dot():
    u, v = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
    ours = basic_ops.dot_product(u, v)
    theirs = float(np.dot(u, v))
    
    print(f"  Dot Product: {u} · {v}")
    print(f"    - Ours:  {ours}")
    print(f"    - NumPy: {theirs}")
    assert np.isclose(ours, theirs)
    print("    ✓ VERIFIED MATCH\n")

def test_cross():
    u, v = [1, 2, 3], [4, 5, 6]
    ours = basic_ops.cross_product_3d(u, v)
    theirs = np.cross(u, v).tolist()
    
    print(f"  Cross Product (3D): {u} × {v}")
    print(f"    - Ours:  {ours}")
    print(f"    - NumPy: {theirs}")
    assert np.allclose(ours, theirs)
    print("    ✓ VERIFIED MATCH\n")

if __name__ == "__main__":
    print("=" * 60)
    print("VERIFICATION: UNIT 01 — BASIC OPS")
    print("=" * 60)
    test_add()
    test_subtract()
    test_dot()
    test_cross()
    print("ALL BASIC OPS VERIFIED ✓")
