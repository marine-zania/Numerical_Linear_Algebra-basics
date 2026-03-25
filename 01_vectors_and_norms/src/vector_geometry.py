"""Vector geometry: angles and projections."""
import math
import vector_ops
import vector_norms

def angle(u, v):
    """Angle in radians between u and v."""
    dot = vector_ops.dot_product(u, v)
    mags = vector_norms.l2(u) * vector_norms.l2(v)
    # Clamp for numerical stability
    cos_theta = max(-1.0, min(1.0, dot / mags))
    return math.acos(cos_theta)

def project(v, u):
    """Project v onto u."""
    coeff = vector_ops.dot_product(v, u) / vector_ops.dot_product(u, u)
    return vector_ops.scalar_multiply(coeff, u)

def decompose(v, u):
    """Decompose v into (parallel, perpendicular) components relative to u."""
    v_parallel = project(v, u)
    v_perp = vector_ops.subtract(v, v_parallel)
    return v_parallel, v_perp
