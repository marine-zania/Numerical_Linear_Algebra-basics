"""Norm implementations and normalization."""

def l1(v):
    """L1 (Manhattan) norm."""
    return sum(abs(x) for x in v)

def l2(v):
    """L2 (Euclidean) norm."""
    return sum(x**2 for x in v)**0.5

def linf(v):
    """L-infinity norm."""
    return max(abs(x) for x in v)

def lp(v, p):
    """General Lp norm."""
    assert p >= 1, "p must be >= 1"
    return sum(abs(x)**p for x in v)**(1.0/p)

def normalize(v, norm_type='l2', p=2):
    """Returns unit vector. norm_type: 'l1', 'l2', 'linf', 'lp'"""
    if norm_type == 'l1': n = l1(v)
    elif norm_type == 'l2': n = l2(v)
    elif norm_type == 'linf': n = linf(v)
    elif norm_type == 'lp': n = lp(v, p)
    else: raise ValueError("Unknown norm type")
    
    assert n > 0, "Cannot normalize zero vector"
    return [x/n for x in v]
