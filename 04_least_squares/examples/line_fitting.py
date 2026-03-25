# Least Squares — Line Fitting Example
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

# Custom Module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.normal_equations import least_squares
from src.fitting import linear_model

def run_example():
    # 0. Find and Load Dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(script_dir, "..", "exercises", "std-rust_test_100000.0.csv")
    
    if not os.path.exists(csv_file):
        print(f"Error: Could not find {csv_file}")
        return
    data = pd.read_csv(csv_file)
    x, y = data["Total Numbers"].values, data["Runtime"].values
    
    print(f"--- Loaded {len(x)} points from {csv_file} ---")

    # 1. Model: y = c0 + c1*x
    # Construct Design Matrix A and Solve
    A = linear_model(x)
    params, residual = least_squares(A, y)
    
    print(f"Intercept (c0): {params[0]:.6e}")
    print(f"Slope (c1):     {params[1]:.6e}")
    print(f"Residual Norm:   {residual:.6f}")

    # 2. Plotting (Save to Outputs folder)
    output_dir = os.path.join(script_dir, "..", "outputs", "line_fitting")
    os.makedirs(output_dir, exist_ok=True)
    
    plt.style.use("dark_background")
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label="Runtime Data", color="aqua", alpha=0.6)
    
    x_line = np.linspace(min(x), max(x), 100)
    y_line = params[0] + params[1] * x_line
    plt.plot(x_line, y_line, label="Linear Fit", color="orangered", linewidth=2)
    
    plt.title("Rust Runtime — Linear Least Squares Fit")
    plt.ylabel("Time (s)")
    plt.xlabel("Total Numbers Generated")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "line-fit-result.png"))
    print(f"Result saved to: {output_dir}")

if __name__ == "__main__":
    run_example()
