# Least Squares — Polynomial Fitting Example
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

# Custom Module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.normal_equations import least_squares
from src.fitting import polynomial_model

def run_example():
    # 0. Find and Load Dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(script_dir, "..", "exercises", "std-rust_test_100000.0.csv")
    
    if not os.path.exists(csv_file):
        print(f"Error: Could not find {csv_file}")
        return
    data = pd.read_csv(csv_file)
    x, y = data["Total Numbers"].values, data["Runtime"].values
    
    # 1. Fit Order 1-4 Models
    for order in range(1, 5):
        print(f"--- Fitting {order}th order polynomial ---")
        A = polynomial_model(x, order)
        params, residual = least_squares(A, y)
        
        for idx, val in enumerate(params):
            print(f"C{idx}: {val:.6e}")
        print(f"Residual Norm: {residual:.6f}\n")
        
        # 2. Plotting (Save to Outputs folder)
        output_dir = os.path.join(script_dir, "..", "outputs", "poly_fitting")
        os.makedirs(output_dir, exist_ok=True)
        
        plt.style.use("dark_background")
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, label="Runtime Data", color="aqua", alpha=0.5)
        
        # Generate line for fit
        x_line = np.linspace(min(x), max(x), 200)
        y_line = np.zeros_like(x_line)
        for i, c in enumerate(params):
            y_line += c * (x_line ** i)
            
        plt.plot(x_line, y_line, label=f"{order}th Order Fit", color="orangered", linewidth=2)
        plt.title(f"{order}th Order Polynomial Rust Runtime Fit")
        plt.ylabel("Time (s)")
        plt.xlabel("Total Numbers Generated")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"poly-fit-order{order}.png"))
        plt.close() # Close to avoid memory issues
        
    print(f"All Result saved to: {output_dir}")

if __name__ == "__main__":
    run_example()