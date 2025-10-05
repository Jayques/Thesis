import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# --- 1. Load COVID-19 Data (Same as Section 5) ---
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

# Select meaningful pairs (same as Section 5)
meaningful_pairs = [
    (0, 1),  # Temperature vs Oxygen Saturation
    (0, 2),  # Temperature vs Age
    (1, 2),  # Oxygen Saturation vs Age
    (0, 3),  # Temperature vs Cough Severity
    (1, 3),  # Oxygen Saturation vs Cough Severity
]

# Combine all data with labels
X_all = np.vstack((covid_positive, covid_negative))
y_all = np.hstack((np.ones(len(covid_positive)), -np.ones(len(covid_negative))))

# --- 2. Create Subplots for Each Feature Pair ---
fig, axes = plt.subplots(2, 3, figsize=(20, 14))
fig.suptitle('COVID-19 Classification using Quadratic Programming (SVM)\n'
             'Pairwise Feature Analysis with Maximum Margin Separation', 
             fontsize=16, fontweight='bold')
axes = axes.ravel()

# Store results for summary table
results_summary = []

# Process each meaningful pair
for idx, (i, j) in enumerate(meaningful_pairs):
    ax = axes[idx]
    
    # Extract the two features for this pair
    positive_pair = covid_positive[:, [i, j]]
    negative_pair = covid_negative[:, [i, j]]
    X_pair = np.vstack((positive_pair, negative_pair))
    
    # Scale the data for these two features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pair)
    
    # --- 3. Define and Solve QP Problem for this Feature Pair ---
    C = 1.0  # Regularization parameter
    
    # Define optimization variables
    w = cp.Variable(2)          # Weight vector [w_i, w_j]
    gamma = cp.Variable()       # Intercept term
    slack = cp.Variable(len(X_scaled))  # Slack variables
    
    # Define constraints
    constraints = [
        y_all[k] * (X_scaled[k] @ w + gamma) >= 1 - slack[k] 
        for k in range(len(X_scaled))
    ]
    constraints.append(slack >= 0)
    
    # Define quadratic objective
    objective = cp.Minimize((1/2) * cp.sum_squares(w) + C * cp.sum_squares(slack))
    
    # Solve the problem
    prob = cp.Problem(objective, constraints)
    
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
    except:
        try:
            prob.solve(solver=cp.SCS, verbose=False)
        except:
            prob.solve(verbose=False)
    
    # Extract optimal values
    w_opt = w.value
    gamma_opt = gamma.value
    
    # --- 4. Calculate Performance Metrics ---
    # Predictions
    predictions = X_scaled @ w_opt + gamma_opt
    predicted_labels = np.sign(predictions)
    accuracy = np.mean(predicted_labels == y_all) * 100
    
    # Margin width
    margin_width = 2 / np.linalg.norm(w_opt) if np.linalg.norm(w_opt) > 1e-10 else 0
    
    # Store results for summary table
    results_summary.append({
        'feature_pair': (i, j),
        'w1': w_opt[0],
        'w2': w_opt[1],
        'gamma': gamma_opt,
        'accuracy': accuracy,
        'margin_width': margin_width,
        'status': prob.status
    })
    
    # --- 5. Plot the Results ---
    # Plot original (unscaled) data points
    ax.scatter(positive_pair[:, 0], positive_pair[:, 1], c='red', marker='o', 
               label='COVID Positive', s=100, edgecolors='black', alpha=0.8)
    ax.scatter(negative_pair[:, 0], negative_pair[:, 1], c='blue', marker='x', 
               label='COVID Negative', s=100, linewidths=2, alpha=0.8)
    
    # Create grid for decision boundary
    x_min = min(np.min(positive_pair[:, 0]), np.min(negative_pair[:, 0]))
    x_max = max(np.max(positive_pair[:, 0]), np.max(negative_pair[:, 0]))
    y_min = min(np.min(positive_pair[:, 1]), np.min(negative_pair[:, 1]))
    y_max = max(np.max(positive_pair[:, 1]), np.max(negative_pair[:, 1]))
    
    x_range = np.linspace(x_min, x_max, 100)
    
    # Calculate decision boundary in original coordinates
    if abs(w_opt[1]) > 1e-10:
        # Transform x_range to scaled coordinates
        x_range_scaled = (x_range - scaler.mean_[0]) / scaler.scale_[0]
        
        # Calculate boundary in scaled coordinates
        y_boundary_scaled = (-w_opt[0] * x_range_scaled - gamma_opt) / w_opt[1]
        
        # Transform back to original coordinates
        y_boundary = y_boundary_scaled * scaler.scale_[1] + scaler.mean_[1]
        
        ax.plot(x_range, y_boundary, 'g-', linewidth=2, label='Decision Boundary')
        
        # Plot margin boundaries
        y_margin_upper_scaled = (1 - w_opt[0] * x_range_scaled - gamma_opt) / w_opt[1]
        y_margin_lower_scaled = (-1 - w_opt[0] * x_range_scaled - gamma_opt) / w_opt[1]
        
        y_margin_upper = y_margin_upper_scaled * scaler.scale_[1] + scaler.mean_[1]
        y_margin_lower = y_margin_lower_scaled * scaler.scale_[1] + scaler.mean_[1]
        
        ax.plot(x_range, y_margin_upper, 'g--', linewidth=1, alpha=0.7, label='Margin')
        ax.plot(x_range, y_margin_lower, 'g--', linewidth=1, alpha=0.7)
    
    # Set labels and title
    ax.set_xlabel(feature_names[i], fontsize=12)
    ax.set_ylabel(feature_names[j], fontsize=12)
    ax.set_title(f'{feature_names[i]} vs {feature_names[j]}\n'
                 f'Accuracy: {accuracy:.1f}%, Margin: {margin_width:.3f}', 
                 fontsize=13)
    ax.grid(True, alpha=0.3)
    
    if idx == 0:
        ax.legend()

# Remove the empty subplot
axes[-1].set_visible(False)

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()

# --- 6. Print Detailed Results Summary ---
print("=" * 80)
print("COVID-19 CLASSIFICATION RESULTS SUMMARY")
print("Quadratic Programming (SVM) Approach")
print("=" * 80)

print("\nFeature Pair Analysis:")
print("-" * 80)
for result in results_summary:
    i, j = result['feature_pair']
    print(f"\n{feature_names[i]} vs {feature_names[j]}:")
    print(f"  w₁ = {result['w1']:.6f}")
    print(f"  w₂ = {result['w2']:.6f}")
    print(f"  γ = {result['gamma']:.6f}")
    print(f"  Accuracy = {result['accuracy']:.1f}%")
    print(f"  Margin Width = {result['margin_width']:.4f}")
    print(f"  Solver Status = {result['status']}")

# --- 7. Create Performance Comparison Table ---
print("\n" + "=" * 80)
print("PERFORMANCE COMPARISON ACROSS FEATURE PAIRS")
print("=" * 80)
print(f"{'Feature Pair':<30} {'Accuracy':<12} {'Margin Width':<15}")
print("-" * 80)

for result in sorted(results_summary, key=lambda x: x['accuracy'], reverse=True):
    i, j = result['feature_pair']
    pair_name = f"{feature_names[i]} - {feature_names[j]}"
    print(f"{pair_name:<30} {result['accuracy']:<10.1f}% {result['margin_width']:<14.4f}")

# --- 8. Identify Best Performing Feature Pair ---
best_result = max(results_summary, key=lambda x: x['accuracy'])
i, j = best_result['feature_pair']
print(f"\nBEST PERFORMING PAIR: {feature_names[i]} - {feature_names[j]}")
print(f"Accuracy: {best_result['accuracy']:.1f}%")
print(f"Decision Rule: {best_result['w1']:.4f}·{feature_names[i]} + "
      f"{best_result['w2']:.4f}·{feature_names[j]} + {best_result['gamma']:.4f} = 0")

# --- 9. Clinical Interpretation ---
print("\n" + "=" * 80)
print("CLINICAL INTERPRETATION")
print("=" * 80)
print("Positive weights indicate higher values push toward COVID Positive classification")
print("Negative weights indicate higher values push toward COVID Negative classification")
print("\nKey Insights:")
for result in results_summary:
    i, j = result['feature_pair']
    w1_sign = "Positive" if result['w1'] > 0 else "Negative"
    w2_sign = "Positive" if result['w2'] > 0 else "Negative"
    
    print(f"- {feature_names[i]}-{feature_names[j]}: {w1_sign} {feature_names[i]} correlation, "
          f"{w2_sign} {feature_names[j]} correlation")