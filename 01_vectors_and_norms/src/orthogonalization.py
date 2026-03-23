"""Orthogonalization algorithms."""
from .geometry import project
from .basic_ops import subtract
from .norms import normalize

def gram_schmidt(vectors):
    """Classical Gram-Schmidt orthonormalization."""
    ortho = []
    for v in vectors:
        u = list(v)
        for q in ortho:
            proj = project(u, q)
            u = subtract(u, proj)
        u = normalize(u, norm_type='l2')
        ortho.append(u)
    return ortho
