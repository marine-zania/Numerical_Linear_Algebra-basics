"""Orthogonalization algorithms."""
import vector_geometry
import vector_ops
import vector_norms

def gram_schmidt(vectors):
    """Classical Gram-Schmidt orthonormalization."""
    ortho = []
    for v in vectors:
        u = list(v)
        for q in ortho:
            proj = vector_geometry.project(u, q)
            u = vector_ops.subtract(u, proj)
        u = vector_norms.normalize(u, norm_type='l2')
        ortho.append(u)
    return ortho
