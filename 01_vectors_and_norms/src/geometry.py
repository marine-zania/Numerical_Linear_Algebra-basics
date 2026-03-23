"""Vector geometry: angles and projections."""
import math
from .basic_ops import dot_product, scalar_multiply, subtract
from .norms import l2

def angle(u, v):
    """Angle in radians between u and v."""
    dot = dot_product(u, v)
    mags = l2(u) * l2(v)
    # Clamp for numerical stability
    cos_theta = max(-1.0, min(1.0, dot / mags))
    return math.acos(cos_theta)

def project(v, u):
    """Project v onto u."""
    coeff = dot_product(v, u) / dot_product(u, u)
    return scalar_multiply(coeff, u)

def decompose(v, u):
    """Decompose v into (parallel, perpendicular) components relative to u."""
    v_parallel = project(v, u)
    v_perp = subtract(v, v_parallel)
    return v_parallel, v_perp
