import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# --- Use the EXACT SAME tumor data from Section 2 ---
R1 = np.array([
    [3.3807, 1.1479, 2.2768, 3.1458, 4.0091, 4.3168, 2.4625, 0.6238, 0.6718, 1.1588],
    [3.7832, 1.1443, 3.6643, 1.0959, 4.1817, 1.5749, 0.8847, 1.1299, 2.7722, 2.1298]
])
R2 = np.array([
    [7.0825, 9.2387, 8.1337, 7.9738, 9.6274, 6.7863, 8.9074, 8.8918, 7.2120, 8.0552],
    [5.8413, 5.7428, 7.8886, 9.0063, 9.7030, 6.0846, 8.0597, 7.6123, 5.5536, 7.0171]
])
M = R1.T  # Malignant tumors (Class +1)
B = R2.T  # Benign tumors (Class -1)

# Combine and label data for easier handling
X = np.vstack((M, B))  # All data points
y = np.hstack((np.ones(M.shape[0]), -np.ones(B.shape[0]))) # Labels: +1 for Malignant, -1 for Benign

# --- Define and Solve the QP Problem ---
n_features = X.shape[1]
C = 10.0  # Regularization parameter. Tune this for different results.

# Define variables
w = cp.Variable(n_features)  # Weight vector [w1, w2]
gamma = cp.Variable()        # Intercept (bias term)
slack = cp.Variable(len(X))  # Slack variables for all data points

# Define constraints
# y_i * (w^T x_i + gamma) >= 1 - slack_i
# This is the standard SVM constraint, equivalent to the two sets of constraints in the LP.
constraints = [y[i] * (X[i] @ w + gamma) >= 1 - slack[i] for i in range(len(X))]
constraints.append(slack >= 0)  # Slack must be non-negative

# Define objective: Minimize (1/2)*||w||^2 + C * (sum of squared slacks)
# The '1/2' is for mathematical convenience when taking the derivative.
objective = cp.Minimize((1/2) * cp.sum_squares(w) + C * cp.sum_squares(slack))

# Solve the problem
prob = cp.Problem(objective, constraints)
prob.solve(verbose=False)  # Remove the solver specification entirely

# Extract the optimal values
w_opt = w.value
gamma_opt = gamma.value
print("QP (SVM) Solution:")
print(f"w = {w_opt}")
print(f"gamma = {gamma_opt:.6f}")
print(f"Solver Status: {prob.status}")

# --- Plot the Results ---
plt.figure(figsize=(10, 6))
# Plot data points
plt.scatter(M[:, 0], M[:, 1], c='red', marker='o', label='Malignant', s=100, edgecolors='black')
plt.scatter(B[:, 0], B[:, 1], c='blue', marker='x', label='Benign', s=100, linewidths=2)

# Plot the separating line (decision boundary: w^T*x + gamma = 0)
x_plot = np.linspace(0, 10, 100)
y_plot = (-w_opt[0] * x_plot - gamma_opt) / w_opt[1]
plt.plot(x_plot, y_plot, 'g-', label='QP Separation Line', linewidth=2)

# Plot the margin lines (w^T*x + gamma = +1 and -1)
margin_upper = (1 - w_opt[0] * x_plot - gamma_opt) / w_opt[1]
margin_lower = (-1 - w_opt[0] * x_plot - gamma_opt) / w_opt[1]
plt.plot(x_plot, margin_upper, 'g--', linewidth=1, label='Margin')
plt.plot(x_plot, margin_lower, 'g--', linewidth=1)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Tumor Classification using Quadratic Programming (SVM)')
plt.legend()
plt.grid(True)
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.show()