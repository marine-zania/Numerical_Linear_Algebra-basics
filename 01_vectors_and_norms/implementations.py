"""
Vectors & Norms — Core Implementations
All operations implemented from scratch (loops / basic math),
then verified against numpy equivalents.
"""

import numpy as np

# 1. Basic Vector Operations (from scratch)

def vector_add(u, v):
    """Element-wise addition of two vectors."""
    assert len(u) == len(v), "Vectors must have the same length"
    return [u[i] + v[i] for i in range(len(u))]


def vector_subtract(u, v):
    """Element-wise subtraction: u - v"""
    assert len(u) == len(v), "Vectors must have the same length"
    return [u[i] - v[i] for i in range(len(u))]


def scalar_multiply(alpha, v):
    """Multiply every element of v by scalar alpha."""
    return [alpha * v[i] for i in range(len(v))]


def dot_product(u, v):
    """Dot (inner) product of two vectors, computed with a loop."""
    assert len(u) == len(v), "Vectors must have the same length"
    result = 0.0
    for i in range(len(u)):
        result += u[i] * v[i]
    return result


def cross_product_3d(u, v):
    """Cross product — only defined for 3D vectors."""
    assert len(u) == 3 and len(v) == 3, "Cross product is only defined in 3D"
    return [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ]


# 2. Norms (from scratch)
def l1_norm(v):
    """L1 norm (Manhattan): sum of absolute values."""
    return sum(abs(xi) for xi in v)


def l2_norm(v):
    """L2 norm (Euclidean): sqrt of sum of squares."""
    return sum(xi ** 2 for xi in v) ** 0.5


def linf_norm(v):
    """L-infinity norm (Chebyshev): max absolute value."""
    return max(abs(xi) for xi in v)


def lp_norm(v, p):
    """General Lp norm: (sum |xi|^p)^(1/p)."""
    assert p >= 1, "p must be >= 1"
    return sum(abs(xi) ** p for xi in v) ** (1.0 / p)


# 3. Normalization
def normalize(v, p=2):
    """Return a unit vector in the direction of v under the Lp norm."""
    norm_val = lp_norm(v, p)
    assert norm_val > 0, "Cannot normalize the zero vector"
    return [xi / norm_val for xi in v]


# 4. Angle Between Vectors
def angle_between(u, v):
    """
    Angle (in radians) between two vectors using:
        cos(theta) = (u · v) / (‖u‖ · ‖v‖)
    """
    import math
    dot = dot_product(u, v)
    norms = l2_norm(u) * l2_norm(v)
    assert norms > 0, "Cannot compute angle with a zero vector"
    # Clamp for numerical safety
    cos_theta = max(-1.0, min(1.0, dot / norms))
    return math.acos(cos_theta)


# 5. Projection
def project(v, u):
    """
    Project vector v onto vector u:
        proj_u(v) = ((v · u) / (u · u)) * u
    """
    coeff = dot_product(v, u) / dot_product(u, u)
    return scalar_multiply(coeff, u)


# 6. Gram-Schmidt Orthogonalization
def gram_schmidt(vectors):
    """
    Given a list of linearly independent vectors, return an orthonormal set.
    Uses the classical Gram-Schmidt process.

    Parameters
    ----------
    vectors : list of lists
        Each inner list is a vector (all same length).

    Returns
    -------
    list of lists
        Orthonormal vectors.
    """
    ortho = []
    for v in vectors:
        # Subtract projections onto all previously computed orthonormal vectors
        u = list(v)  # copy
        for q in ortho:
            proj = project(u, q)
            u = vector_subtract(u, proj)
        # Normalize
        u = normalize(u, p=2)
        ortho.append(u)
    return ortho


# Verification against NumPy
def verify_all():
    """Run all implementations and compare with numpy."""
    print("=" * 60)
    print("VECTORS & NORMS — Verification")
    print("=" * 60)

    u = [1.0, -2.0, 3.0, 0.5]
    v = [4.0, 0.0, -1.0, 2.0]
    u_np, v_np = np.array(u), np.array(v)

    # --- Basic operations ---
    print("\n--- Basic Operations ---")

    result = vector_add(u, v)
    expected = u_np + v_np
    print(f"  add:       {result}  (numpy: {expected.tolist()})")
    assert np.allclose(result, expected)

    result = vector_subtract(u, v)
    expected = u_np - v_np
    print(f"  subtract:  {result}  (numpy: {expected.tolist()})")
    assert np.allclose(result, expected)

    result = scalar_multiply(3.0, u)
    expected = 3.0 * u_np
    print(f"  scalar*3:  {result}  (numpy: {expected.tolist()})")
    assert np.allclose(result, expected)

    result = dot_product(u, v)
    expected = np.dot(u_np, v_np)
    print(f"  dot:       {result}  (numpy: {expected})")
    assert np.isclose(result, expected)

    # --- Cross product (3D only) ---
    u3, v3 = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
    result = cross_product_3d(u3, v3)
    expected = np.cross(u3, v3).tolist()
    print(f"  cross:     {result}  (numpy: {expected})")
    assert np.allclose(result, expected)

    # --- Norms ---
    print("\n--- Norms ---")

    result = l1_norm(u)
    expected = np.linalg.norm(u_np, 1)
    print(f"  L1:        {result:.6f}  (numpy: {expected:.6f})")
    assert np.isclose(result, expected)

    result = l2_norm(u)
    expected = np.linalg.norm(u_np, 2)
    print(f"  L2:        {result:.6f}  (numpy: {expected:.6f})")
    assert np.isclose(result, expected)

    result = linf_norm(u)
    expected = np.linalg.norm(u_np, np.inf)
    print(f"  Linf:      {result:.6f}  (numpy: {expected:.6f})")
    assert np.isclose(result, expected)

    result = lp_norm(u, 3)
    expected = np.linalg.norm(u_np, 3)
    print(f"  L3:        {result:.6f}  (numpy: {expected:.6f})")
    assert np.isclose(result, expected)

    # --- Normalization ---
    print("\n--- Normalization ---")
    u_hat = normalize(u, p=2)
    print(f"  unit vec:  {[f'{x:.4f}' for x in u_hat]}")
    print(f"  ‖û‖₂ = {l2_norm(u_hat):.10f}  (should be 1.0)")
    assert np.isclose(l2_norm(u_hat), 1.0)

    # --- Angle ---
    print("\n--- Angle Between Vectors ---")
    import math
    # Orthogonal vectors
    a, b = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
    theta = angle_between(a, b)
    print(f"  orthogonal: {math.degrees(theta):.1f}°  (should be 90.0°)")
    assert np.isclose(theta, math.pi / 2)

    # Parallel vectors
    a, b = [1.0, 2.0, 3.0], [2.0, 4.0, 6.0]
    theta = angle_between(a, b)
    print(f"  parallel:   {math.degrees(theta):.1f}°  (should be 0.0°)")
    assert np.isclose(theta, 0.0)

    # Anti-parallel
    a, b = [1.0, 0.0], [-1.0, 0.0]
    theta = angle_between(a, b)
    print(f"  anti-par:   {math.degrees(theta):.1f}°  (should be 180.0°)")
    assert np.isclose(theta, math.pi)

    # --- Projection ---
    print("\n--- Projection ---")
    v_proj = project([3.0, 4.0], [1.0, 0.0])
    print(f"  proj of [3,4] onto [1,0]: {v_proj}  (should be [3.0, 0.0])")
    assert np.allclose(v_proj, [3.0, 0.0])

    # Verify residual is orthogonal to u
    residual = vector_subtract([3.0, 4.0], v_proj)
    dot_check = dot_product(residual, [1.0, 0.0])
    print(f"  residual · u = {dot_check:.10f}  (should be 0.0)")
    assert np.isclose(dot_check, 0.0)

    # --- Gram-Schmidt ---
    print("\n--- Gram-Schmidt ---")
    vecs = [[1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]
    ortho = gram_schmidt(vecs)
    print(f"  Input:  {vecs}")
    for i, q in enumerate(ortho):
        print(f"  q{i}: [{', '.join(f'{x:.4f}' for x in q)}]  ‖q‖={l2_norm(q):.6f}")

    # Verify orthonormality
    n = len(ortho)
    for i in range(n):
        assert np.isclose(l2_norm(ortho[i]), 1.0), f"q{i} is not unit length"
        for j in range(i + 1, n):
            d = dot_product(ortho[i], ortho[j])
            assert np.isclose(d, 0.0, atol=1e-12), f"q{i} · q{j} = {d} ≠ 0"
    print("  ✓ All pairs orthogonal, all unit length")

    print("\n" + "=" * 60)
    print("ALL VERIFICATIONS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    verify_all()
