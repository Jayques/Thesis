import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Generate the Synthetic Vital Signs Data (Same as Section 3) ---
np.random.seed(42)
num_patients = 10

# Stable patients: Heart Rate ~ N(75, 10²), Blood Pressure ~ N(120, 10²)
stable_hr = np.random.normal(75, 10, num_patients)
stable_bp = np.random.normal(120, 10, num_patients)
stable_data = np.column_stack((stable_hr, stable_bp))

# At-risk patients: Heart Rate ~ N(110, 15²), Blood Pressure ~ N(160, 15²)
at_risk_hr = np.random.normal(110, 15, num_patients)
at_risk_bp = np.random.normal(160, 15, num_patients)
at_risk_data = np.column_stack((at_risk_hr, at_risk_bp))

# --- 2. Prepare Data for QP Formulation ---
# Combine all data points
X = np.vstack((stable_data, at_risk_data))

# Create labels: +1 for Stable, -1 for At-Risk
y = np.hstack((np.ones(num_patients), -np.ones(num_patients)))

# --- 3. Define and Solve the QP Problem ---
n_features = X.shape[1]
C = 1.0  # Regularization parameter - can be tuned

# Define optimization variables
w = cp.Variable(n_features)    # Weight vector [w1, w2]
gamma = cp.Variable()          # Intercept term
slack = cp.Variable(len(X))    # Slack variables for all patients

# Define constraints: y_i*(w^T*x_i + gamma) >= 1 - slack_i
constraints = [
    y[i] * (X[i] @ w + gamma) >= 1 - slack[i] for i in range(len(X))
]
constraints.append(slack >= 0)  # Slack variables must be non-negative

# Define quadratic objective: (1/2)*||w||^2 + C*(sum of squared slacks)
objective = cp.Minimize((1/2) * cp.sum_squares(w) + C * cp.sum_squares(slack))

# Create and solve the problem
prob = cp.Problem(objective, constraints)

# Solve with automatic solver selection
try:
    prob.solve(solver=cp.ECOS, verbose=False)
    print("Using ECOS solver")
except:
    try:
        prob.solve(solver=cp.SCS, verbose=False)
        print("Using SCS solver")
    except:
        prob.solve(verbose=False)
        print("Using default solver")

# --- 4. Extract and Display Results ---
w_opt = w.value
gamma_opt = gamma.value

print("\n=== QP/SVM Medical Triage Classification Results ===")
print(f"Solver Status: {prob.status}")
print(f"Optimal Parameters:")
print(f"w₁ (Heart Rate coeff) = {w_opt[0]:.6f}")
print(f"w₂ (Blood Pressure coeff) = {w_opt[1]:.6f}")
print(f"γ (Intercept) = {gamma_opt:.6f}")
print(f"\nSeparating Hyperplane Equation:")
print(f"{w_opt[0]:.4f}·HR + {w_opt[1]:.4f}·BP + {gamma_opt:.4f} = 0")
print(f"\nClassification Rule:")
print(f"If {w_opt[0]:.4f}·HR + {w_opt[1]:.4f}·BP + {gamma_opt:.4f} >= 0 → Stable")
print(f"If {w_opt[0]:.4f}·HR + {w_opt[1]:.4f}·BP + {gamma_opt:.4f} < 0 → At-Risk")

# --- 5. Plot the Results ---
plt.figure(figsize=(12, 8))

# Plot data points
plt.scatter(stable_hr, stable_bp, c='green', marker='o', label='Stable', 
            s=100, edgecolors='black', alpha=0.8)
plt.scatter(at_risk_hr, at_risk_bp, c='red', marker='x', label='At-Risk', 
            s=100, linewidths=2, alpha=0.8)

# Plot the separating line (decision boundary: w^T*x + gamma = 0)
hr_range = np.array([50, 140])
bp_range = (-w_opt[0] * hr_range - gamma_opt) / w_opt[1]
plt.plot(hr_range, bp_range, 'b-', label='Decision Boundary', linewidth=2)

# Plot the margin lines (w^T*x + gamma = +1 and -1)
margin_upper = (1 - w_opt[0] * hr_range - gamma_opt) / w_opt[1]
margin_lower = (-1 - w_opt[0] * hr_range - gamma_opt) / w_opt[1]
plt.plot(hr_range, margin_upper, 'b--', linewidth=1, label='Margin')
plt.plot(hr_range, margin_lower, 'b--', linewidth=1)

# Calculate margin width
margin_width = 2 / np.linalg.norm(w_opt)
print(f"\nMargin Width: {margin_width:.4f}")

# Add labels and title
plt.xlabel('Heart Rate (bpm)', fontsize=12)
plt.ylabel('Blood Pressure (mmHg)', fontsize=12)
plt.title('Medical Triage Classification using Quadratic Programming (SVM)\n'
          f'Margin Width: {margin_width:.3f}', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# Set appropriate limits
plt.xlim(40, 180)
plt.ylim(80, 200)

plt.tight_layout()
plt.show()

# --- 6. Performance Evaluation ---
# Predict classes for all patients
predictions = X @ w_opt + gamma_opt
predicted_labels = np.sign(predictions)

# Calculate accuracy
accuracy = np.mean(predicted_labels == y) * 100
print(f"\nClassification Accuracy: {accuracy:.1f}%")

# Show predictions for each patient
print("\nPatient-by-Patient Predictions:")
print("HR\tBP\tActual\tPredicted\tCorrect")
print("-" * 45)
for i in range(len(X)):
    correct = "✓" if predicted_labels[i] == y[i] else "✗"
    print(f"{X[i,0]:.1f}\t{X[i,1]:.1f}\t{'Stable' if y[i] == 1 else 'At-Risk'}\t"
          f"{'Stable' if predicted_labels[i] == 1 else 'At-Risk'}\t\t{correct}")