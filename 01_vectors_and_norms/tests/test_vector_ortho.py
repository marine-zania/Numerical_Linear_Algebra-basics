"""Loud & Clear: Orthogonalization Verification."""
import numpy as np
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '..', 'src'))
import vector_ortho, vector_ops, vector_norms

def test_gram_schmidt():
    vecs = [[1, 1, 0], [1, 0, 1], [0, 1, 1]]
    print(f"  Gram-Schmidt on Vectors: {vecs}")
    
    ortho = vector_ortho.gram_schmidt(vecs)
    
    # 1. Verification of Orthonormality
    print("\n    Verifying Orthonormality Results:")
    for i, q in enumerate(ortho):
        mag = vector_norms.l2(q)
        print(f"    - q{i} magnitude: {mag:.6f} (Should be 1.0)")
        assert np.isclose(mag, 1.0)
        
        for j in range(i + 1, len(ortho)):
            dot = vector_ops.dot_product(q, ortho[j])
            print(f"    - Cross check q{i}·q{j}: {dot:+.12f} (Should be 0.0)")
            assert np.isclose(dot, 0.0, atol=1e-12)
            
    print("\n    ✓ VERIFIED: Results are Orthonormal\n")

if __name__ == "__main__":
    print("=" * 60)
    print("VERIFICATION: UNIT 01 — ORTHOGONALIZATION")
    print("=" * 60)
    test_gram_schmidt()
    print("ORTHOGONALIZATION VERIFIED ✓")
