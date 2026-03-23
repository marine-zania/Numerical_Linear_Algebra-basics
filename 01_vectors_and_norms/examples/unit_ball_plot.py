"""
Example: Unit ball visualization in 2D
Uses norms.lp implemented in src.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src import norms

def plot_unit_balls():
    x = np.linspace(-1.5, 1.5, 400)
    y = np.linspace(-1.5, 1.5, 400)
    X, Y = np.meshgrid(x, y)
    
    # Grid of vectors as tuples (x, y)
    Z_l1 = np.abs(X) + np.abs(Y)
    Z_l2 = np.sqrt(X**2 + Y**2)
    Z_linf = np.maximum(np.abs(X), np.abs(Y))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, (name, Z) in zip(axes, [("L1", Z_l1), ("L2", Z_l2), ("L∞", Z_linf)]):
        ax.contourf(X, Y, Z, levels=[0, 1.0], colors=["#3b82f6"], alpha=0.3)
        ax.contour(X, Y, Z, levels=[1.0], colors=["#1d4ed8"], linewidths=2)
        ax.set_aspect("equal")
        ax.set_title(name, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Unit Balls: Visualizing Norm Boundaries", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig("unit_balls.png", dpi=150)
    print("Saved unit_balls.png")

if __name__ == "__main__":
    plot_unit_balls()
