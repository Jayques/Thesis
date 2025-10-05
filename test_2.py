import pulp as lp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import itertools

# Example COVID-19 data with 4 features
# Format: [temperature, oxygen_saturation, age, cough_severity]
covid_positive = np.array([
    [38.5, 92, 65, 3],
    [37.8, 94, 72, 2],
    [39.1, 89, 58, 3],
    [38.2, 93, 61, 2],
    [37.9, 95, 45, 1],
    [38.7, 91, 70, 3],
    [38.0, 96, 52, 1],
    [39.2, 88, 68, 3]
])

covid_negative = np.array([
    [36.8, 98, 32, 0],
    [36.5, 99, 28, 0],
    [37.1, 97, 41, 1],
    [36.9, 98, 35, 0],
    [37.0, 99, 29, 0],
    [36.7, 98, 38, 1],
    [37.2, 97, 44, 1],
    [36.6, 99, 31, 0]
])

feature_names = ['Temperature', 'Oxygen Saturation', 'Age', 'Cough Severity']

# Select meaningful pairs - these make clinical sense to analyze together
meaningful_pairs = [
    (0, 1),  # Temperature vs Oxygen Saturation
    (0, 2),  # Temperature vs Age
    (1, 2),  # Oxygen Saturation vs Age
    (0, 3),  # Temperature vs Cough Severity
    (1, 3),  # Oxygen Saturation vs Cough Severity
]

# Create a figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Pairwise COVID-19 Classification with Individual LP Models', fontsize=16)
axes = axes.ravel()  # Flatten the axes array for easier indexing

# Process each meaningful pair
for idx, (i, j) in enumerate(meaningful_pairs):
    ax = axes[idx]
    
    # Extract the two features for this pair
    positive_pair = covid_positive[:, [i, j]]
    negative_pair = covid_negative[:, [i, j]]
    
    # Scale the data for these two features
    scaler = StandardScaler()
    positive_scaled = scaler.fit_transform(positive_pair)
    negative_scaled = scaler.transform(negative_pair)
    
    # Create a Linear Programming model for this feature pair
    prob = lp.LpProblem(f"COVID_Classification_{feature_names[i]}_{feature_names[j]}", lp.LpMinimize)
    
    # Get dimensions
    m = positive_scaled.shape[0]  # Number of positive cases
    k = negative_scaled.shape[0]  # Number of negative cases
    n = 2  # Number of features in this pair
    
    # Define the Decision Variables
    w1 = lp.LpVariable("w1", cat="Continuous")
    w2 = lp.LpVariable("w2", cat="Continuous")
    gamma = lp.LpVariable("gamma", cat="Continuous")
    y = [lp.LpVariable(f"y_{i}", lowBound=0) for i in range(m)]  # Violations for positive cases
    z = [lp.LpVariable(f"z_{i}", lowBound=0) for i in range(k)]  # Violations for negative cases
    
    # Define the Objective Function
    prob += (lp.lpSum(y) / m) + (lp.lpSum(z) / k)
    
    # Add the Constraints for positive cases
    for idx_p in range(m):
        prob += y[idx_p] >= 1 - (w1 * positive_scaled[idx_p, 0] + w2 * positive_scaled[idx_p, 1] - gamma)
    
    # Add the Constraints for negative cases
    for idx_n in range(k):
        prob += z[idx_n] >= (w1 * negative_scaled[idx_n, 0] + w2 * negative_scaled[idx_n, 1] - gamma) + 1
    
    # Solve the Problem
    solver = lp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)
    
    # Extract the optimal values
    w1_opt = w1.varValue
    w2_opt = w2.varValue
    gamma_opt = gamma.varValue
    
    # Plot the data points
    ax.scatter(positive_pair[:, 0], positive_pair[:, 1], c='red', marker='o', 
              label='COVID Positive', s=100, edgecolors='black', alpha=0.7)
    ax.scatter(negative_pair[:, 0], negative_pair[:, 1], c='blue', marker='x', 
              label='COVID Negative', s=100, linewidths=2, alpha=0.7)
    
    # Create a grid for the decision boundary
    x_min = min(np.min(positive_pair[:, 0]), np.min(negative_pair[:, 0]))
    x_max = max(np.max(positive_pair[:, 0]), np.max(negative_pair[:, 0]))
    y_min = min(np.min(positive_pair[:, 1]), np.min(negative_pair[:, 1]))
    y_max = max(np.max(positive_pair[:, 1]), np.max(negative_pair[:, 1]))
    
    # Extend the range a bit for better visualization
    x_range = np.linspace(x_min - 0.1*(x_max-x_min), x_max + 0.1*(x_max-x_min), 100)
    
    # Calculate the decision boundary with a more robust approach
    # Check if w2 is close to zero to avoid division by zero
    if abs(w2_opt) < 1e-10:
        # If w2 is nearly zero, the boundary is a vertical line
        boundary_x = np.full_like(x_range, gamma_opt / w1_opt if abs(w1_opt) > 1e-10 else 0)
        boundary_y = np.linspace(y_min, y_max, 100)
        
        # Plot vertical line
        ax.axvline(x=boundary_x[0], color='green', linewidth=2, label='Decision Boundary')
        
        # Plot margin boundaries
        margin = 0.5
        ax.axvline(x=boundary_x[0] + margin/w1_opt if abs(w1_opt) > 1e-10 else 0, 
                  color='green', linestyle='--', linewidth=1, alpha=0.7, label='Margin')
        ax.axvline(x=boundary_x[0] - margin/w1_opt if abs(w1_opt) > 1e-10 else 0, 
                  color='green', linestyle='--', linewidth=1, alpha=0.7)
    else:
        # Normal case: calculate y values for the boundary
        # We need to scale the x values first
        x_range_scaled = (x_range - scaler.mean_[0]) / scaler.scale_[0]
        
        # Calculate the boundary in the scaled space
        boundary_y_scaled = (gamma_opt - w1_opt * x_range_scaled) / w2_opt
        
        # Transform back to the original space
        boundary_y = boundary_y_scaled * scaler.scale_[1] + scaler.mean_[1]
        
        # Filter out extreme values that might cause issues
        valid_indices = (boundary_y >= y_min - 0.5*(y_max-y_min)) & (boundary_y <= y_max + 0.5*(y_max-y_min))
        x_range_filtered = x_range[valid_indices]
        boundary_y_filtered = boundary_y[valid_indices]
        
        # Plot the decision boundary
        ax.plot(x_range_filtered, boundary_y_filtered, 'g-', linewidth=2, label='Decision Boundary')
        
        # Add margin boundaries
        margin = 0.5
        boundary_y_scaled_upper = (gamma_opt + margin - w1_opt * x_range_scaled) / w2_opt
        boundary_y_upper = boundary_y_scaled_upper * scaler.scale_[1] + scaler.mean_[1]
        
        boundary_y_scaled_lower = (gamma_opt - margin - w1_opt * x_range_scaled) / w2_opt
        boundary_y_lower = boundary_y_scaled_lower * scaler.scale_[1] + scaler.mean_[1]
        
        # Filter extreme values
        valid_upper = (boundary_y_upper >= y_min - 0.5*(y_max-y_min)) & (boundary_y_upper <= y_max + 0.5*(y_max-y_min))
        valid_lower = (boundary_y_lower >= y_min - 0.5*(y_max-y_min)) & (boundary_y_lower <= y_max + 0.5*(y_max-y_min))
        
        ax.plot(x_range[valid_upper], boundary_y_upper[valid_upper], 
                'g--', linewidth=1, alpha=0.7, label='Margin')
        ax.plot(x_range[valid_lower], boundary_y_lower[valid_lower], 
                'g--', linewidth=1, alpha=0.7)
    
    # Set labels and title
    ax.set_xlabel(feature_names[i])
    ax.set_ylabel(feature_names[j])
    ax.set_title(f'{feature_names[i]} vs {feature_names[j]}')
    ax.grid(True, alpha=0.3)
    
    # Set appropriate axis limits
    ax.set_xlim(x_min - 0.1*(x_max-x_min), x_max + 0.1*(x_max-x_min))
    ax.set_ylim(y_min - 0.1*(y_max-y_min), y_max + 0.1*(y_max-y_min))
    
    # Add legend to the first plot only
    if idx == 0:
        ax.legend()
    
    # Print model information
    print(f"Model for {feature_names[i]} vs {feature_names[j]}:")
    print(f"  w1 = {w1_opt:.4f}, w2 = {w2_opt:.4f}, gamma = {gamma_opt:.4f}")
    print(f"  Objective value: {lp.value(prob.objective):.4f}")
    print()

# Remove the empty subplot if we have an odd number of pairs
if len(meaningful_pairs) < 6:
    axes[-1].set_visible(False)

plt.tight_layout()
plt.show()

# Create a summary table of the models
print("Summary of LP Models for Each Feature Pair:")
print("=" * 70)
print(f"{'Feature Pair':<25} {'w1':<10} {'w2':<10} {'gamma':<10} {'Objective':<10}")
print("-" * 70)

# Re-run a quick version to get all the values for the summary
for i, j in meaningful_pairs:
    # Extract the two features
    positive_pair = covid_positive[:, [i, j]]
    negative_pair = covid_negative[:, [i, j]]
    
    # Scale the data
    scaler = StandardScaler()
    positive_scaled = scaler.fit_transform(positive_pair)
    negative_scaled = scaler.transform(negative_pair)
    
    # Create and solve the LP model
    prob = lp.LpProblem(f"Summary_Model", lp.LpMinimize)
    
    m = positive_scaled.shape[0]
    k = negative_scaled.shape[0]
    
    w1 = lp.LpVariable("w1", cat="Continuous")
    w2 = lp.LpVariable("w2", cat="Continuous")
    gamma = lp.LpVariable("gamma", cat="Continuous")
    y = [lp.LpVariable(f"y_{i}", lowBound=0) for i in range(m)]
    z = [lp.LpVariable(f"z_{i}", lowBound=0) for i in range(k)]
    
    prob += (lp.lpSum(y) / m) + (lp.lpSum(z) / k)
    
    for idx_p in range(m):
        prob += y[idx_p] >= 1 - (w1 * positive_scaled[idx_p, 0] + w2 * positive_scaled[idx_p, 1] - gamma)
    
    for idx_n in range(k):
        prob += z[idx_n] >= (w1 * negative_scaled[idx_n, 0] + w2 * negative_scaled[idx_n, 1] - gamma) + 1
        
    
    solver = lp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)
    
    # Print summary
    pair_name = f"{feature_names[i]} - {feature_names[j]}"
    print(f"{pair_name:<25} {w1.varValue:<10.4f} {w2.varValue:<10.4f} {gamma.varValue:<10.4f} {lp.value(prob.objective):<10.4f}")