import pulp as lp
import numpy as np
import matplotlib.pyplot as plt

# Data from the PDF
R1 = np.array([
    [3.3807, 1.1479, 2.2768, 3.1458, 4.0091, 4.3168, 2.4625, 0.6238, 0.6718, 1.1588],
    [3.7832, 1.1443, 3.6643, 1.0959, 4.1817, 1.5749, 0.8847, 1.1299, 2.7722, 2.1298]
])

R2 = np.array([
    [7.0825, 9.2387, 8.1337, 7.9738, 9.6274, 6.7863, 8.9074, 8.8918, 7.2120, 8.0552],
    [5.8413, 5.7428, 7.8886, 9.0063, 9.7030, 6.0846, 8.0597, 7.6123, 5.5536, 7.0171]
])

# Transpose to get points in rows
M = R1.T  # First set (malignant tumors)
B = R2.T  # Second set (benign tumors)

# Get dimensions
m = M.shape[0]  # Number of points in first set
k = B.shape[0]  # Number of points in second set
n = M.shape[1]  # Number of dimensions

# Create the Linear Programming Problem
prob = lp.LpProblem("Tumor_Classification", lp.LpMinimize)

# Define the Decision Variables
w = [lp.LpVariable(f"w_{i}", cat="Continuous") for i in range(n)]
gamma = lp.LpVariable("gamma", cat="Continuous")
y = [lp.LpVariable(f"y_{i}", lowBound=0) for i in range(m)]  # Violations for first set
z = [lp.LpVariable(f"z_{i}", lowBound=0) for i in range(k)]  # Violations for second set

# Define the Objective Function
prob += (lp.lpSum(y) / m) + (lp.lpSum(z) / k)

# Add the Constraints for the first set (M)
for i in range(m):
    # y_i >= -(w'*M_i - gamma) + 1
    # Which is equivalent to: y_i >= 1 - (w'*M_i - gamma)
    prob += y[i] >= 1 - (lp.lpSum([w[j] * M[i, j] for j in range(n)]) - gamma)

# Add the Constraints for the second set (B)
for i in range(k):
    # z_i >= (w'*B_i - gamma) + 1
    prob += z[i] >= (lp.lpSum([w[j] * B[i, j] for j in range(n)]) - gamma) + 1

# Solve the Problem
solver = lp.PULP_CBC_CMD(msg=False)
prob.solve(solver)

# Print the Results
print("Solver Status:", lp.LpStatus[prob.status])
print("Objective Value (Total Avg. Violation):", lp.value(prob.objective))
print("\nOptimal Parameters:")
for i in range(n):
    print(f"w_{i+1} = {w[i].varValue:.6f}")
print(f"gamma = {gamma.varValue:.6f}")

# Extract the optimal values
w_opt = np.array([var.varValue for var in w])
gamma_opt = gamma.varValue

# Print the separating line equation
print(f"\nSeparating Line: {w_opt[0]:.4f}*x1 + {w_opt[1]:.4f}*x2 = {gamma_opt:.4f}")

# 8. Plot the Results
plt.figure(figsize=(10, 6))

# Plot the first set (M - malignant tumors)
plt.scatter(M[:, 0], M[:, 1], c='red', marker='o', label='Malignant', s=100, edgecolors='black')

# Plot the second set (B - benign tumors)
plt.scatter(B[:, 0], B[:, 1], c='blue', marker='x', label='Benign', s=100, linewidths=2)

# Plot the separating line
x_range = np.array([0, 10])  # X-axis range based on data
# Solve w1*x1 + w2*x2 = gamma for x2
y_range = (gamma_opt - w_opt[0] * x_range) / w_opt[1]
plt.plot(x_range, y_range, 'g-', label='Separating Line', linewidth=2)

# Plot the buffer lines (margin boundaries)
y_range_upper = (gamma_opt + 1 - w_opt[0] * x_range) / w_opt[1]  # w'x = gamma + 1
y_range_lower = (gamma_opt - 1 - w_opt[0] * x_range) / w_opt[1]  # w'x = gamma - 1
plt.plot(x_range, y_range_upper, 'g--', label='Upper Margin', linewidth=1)
plt.plot(x_range, y_range_lower, 'g--', label='Lower Margin', linewidth=1)

plt.xlabel('Feature 1 (x1)')
plt.ylabel('Feature 2 (x2)')
plt.title('Tumor Classification using Linear Programming')
plt.legend()
plt.grid(True)
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.show()