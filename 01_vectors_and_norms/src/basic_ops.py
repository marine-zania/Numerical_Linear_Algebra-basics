"""Basic vector operations implemented from scratch."""

def add(u, v):
    """Element-wise addition: u + v"""
    assert len(u) == len(v), "Vectors must have same length"
    return [u[i] + v[i] for i in range(len(u))]

def subtract(u, v):
    """Element-wise subtraction: u - v"""
    assert len(u) == len(v), "Vectors must have same length"
    return [u[i] - v[i] for i in range(len(u))]

def scalar_multiply(alpha, v):
    """Scalar multiplication: alpha * v"""
    return [alpha * x for x in v]

def dot_product(u, v):
    """Dot product: u · v"""
    assert len(u) == len(v), "Vectors must have same length"
    return sum(u[i] * v[i] for i in range(len(u)))

def cross_product_3d(u, v):
    """Cross product for 3D vectors."""
    assert len(u) == 3 and len(v) == 3, "Cross product defined for 3D only"
    return [
        u[1]*v[2] - u[2]*v[1],
        u[2]*v[0] - u[0]*v[2],
        u[0]*v[1] - u[1]*v[0]
    ]
