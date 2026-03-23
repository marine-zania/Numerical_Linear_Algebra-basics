"""Loud & Clear: Norms Verification."""
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src import norms

def test_l1():
    v = [3, -4, 0, 1.5]
    ours = norms.l1(v)
    theirs = float(np.linalg.norm(v, 1))
    
    print(f"  L1 Norm of {v}")
    print(f"    - Ours:  {ours}")
    print(f"    - NumPy: {theirs}")
    assert np.isclose(ours, theirs)
    print("    ✓ VERIFIED MATCH\n")

def test_l2():
    v = [3, -4, 0, 1.5]
    ours = norms.l2(v)
    theirs = float(np.linalg.norm(v, 2))
    
    print(f"  L2 Norm of {v}")
    print(f"    - Ours:  {ours:.6f}")
    print(f"    - NumPy: {theirs:.6f}")
    assert np.isclose(ours, theirs)
    print("    ✓ VERIFIED MATCH\n")

def test_linf():
    v = [3, -4, 0, 1.5]
    ours = norms.linf(v)
    theirs = float(np.linalg.norm(v, np.inf))
    
    print(f"  Linf Norm of {v}")
    print(f"    - Ours:  {ours}")
    print(f"    - NumPy: {theirs}")
    assert np.isclose(ours, theirs)
    print("    ✓ VERIFIED MATCH\n")

def test_normalization():
    v = [1, 2, 2]
    ours = norms.normalize(v, 'l2')
    # Result should be [1/3, 2/3, 2/3]
    normalized_l2 = np.linalg.norm(ours, 2)
    
    print(f"  L2 Normalization of {v}")
    print(f"    - Result: {ours}")
    print(f"    - Verified L2 Magnitude: {normalized_l2:.6f} (Should be 1.0)")
    assert np.isclose(normalized_l2, 1.0)
    print("    ✓ VERIFIED MATCH\n")

if __name__ == "__main__":
    print("=" * 60)
    print("VERIFICATION: UNIT 01 — NORMS")
    print("=" * 60)
    test_l1()
    test_l2()
    test_linf()
    test_normalization()
    print("ALL NORMS VERIFIED ✓")
